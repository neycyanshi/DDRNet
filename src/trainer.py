import os
import time
import logging
from operator import mul
from functools import reduce
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import ops
import utils
import model
from losses import LOSSES_COLLECTION

class Trainer(object):
    def __init__(self, filename, config):
        """

        :type config: object
        """
        self.params = dict()
        self.config = config

        self._prepare_inputs(filename)
        self.summaries = list()
        self._build_model()

        self.summaries.extend([
            tf.summary.image("depth_raw", self.pseudo_color(self.params["depth_raw"])),
            tf.summary.image("depth_ref", self.pseudo_color(self.params["depth_ref"])),
            tf.summary.image("normals_raw", self.params["normals_raw"]),
            tf.summary.image("normals_ref", self.params["normals_ref"]),
            tf.summary.image("color", self.params["color"]),
            tf.summary.image("mask", 255*tf.cast(self.params["mask"], tf.uint8)),
        ])

        if config.dnnet != "None":
            self.summaries.extend([
                tf.summary.image("depth_dn", self.pseudo_color(self.params["depth_dn"])),
                tf.summary.image("normals_dn", self.params["normals_dn"])
            ])

        if config.dtnet != "None":
            self.summaries.extend([
                tf.summary.image("depth_dt", self.pseudo_color(self.params["depth_dt"])),
                tf.summary.image("normals_dt", self.params["normals_dt"]),
                tf.summary.image("albedo", self.params["albedo"]),
                tf.summary.image("irrad", self.params["irrad"])
            ])
        self.all_summaries = tf.summary.merge(self.summaries)
        self.summary_writer = tf.summary.FileWriter(config.logdir)
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)

        self.sv = tf.train.Supervisor(
            logdir=config.logdir,
            saver=self.saver,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_model_secs=0,
            checkpoint_basename=config.checkpoint_basename,
            global_step=self.params["global_step"])

        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = self.sv.prepare_or_wait_for_session(config=sess_config)


    def _prepare_inputs(self, filename):
        config, params = self.config, self.params

        global_step = tf.Variable(0, trainable=False, name="global_step")
        is_training = tf.placeholder(tf.bool, name="is_training")
        intr, depth_raw, depth_ref, color, mask, abd, self.diff = ops.read_image_from_filename(
            filename,
            has_mask=config.has_mask,
            has_abd=config.has_abd,
            aux_type=config.aux_type,
            low_th=config.low_thres,
            up_th=config.up_thres,
            diff_th=config.diff_thres,
            batch_size=config.batch_size,
            num_threads=config.num_threads,
            output_height=config.image_size,
            output_width=config.image_size,
            min_after_dequeue=config.min_after_dequeue,
            use_shuffle_batch=config.use_shuffle_batch,
            rand_crop=config.rand_crop,
            rand_scale=config.rand_scale,
            rand_flip=config.rand_flip,
            rand_depth_shift=config.rand_depth_shift,
            rand_brightness=config.rand_brightness
        )

        params["is_training"] = is_training
        params["global_step"] = global_step

        params["intr"] = intr
        params["depth_raw"] = depth_raw
        params["depth_ref"] = depth_ref
        params["color"] = color
        params["mask"] = mask
        params["abd"] = abd


    def _build_model(self):
        config, params = self.config, self.params

        is_training = params["is_training"]
        global_step = params["global_step"]

        depth_raw = params["depth_raw"]
        depth_ref = params["depth_ref"]
        intr = params["intr"]
        color = params["color"]

        dnnet = dtnet = None
        if config.dnnet == "None":
            dnnet = None
        elif config.dnnet == "base":
            dnnet = model.base
        elif config.dnnet == "unet":
            dnnet = model.unet
        elif config.dnnet == "convResnet":
            dnnet = model.convResnet
        elif config.dnnet == "UResnet":
            dnnet = model.UResnet
        else:
            raise NotImplementedError("There is no such {} dnnet".format(config.dnnet))
        if config.dtnet == "None":
            dtnet = None
        elif config.dtnet == "hypercolumn":
            dtnet = model.hypercolumn
        else:
            raise NotImplementedError("There is no such {} dtnet".format(config.dtnet))

        if dnnet:
            depth_dn, _, _ = dnnet(depth_raw, is_training, aux=None, scope="dn_net")
            # params["w_cnv0"] = weight_vars[0]
            # params["out_encoder4"] = end_pts['dn_net/encoder_4']
        else:
            depth_dn = depth_raw

        if dtnet:
            depth_dt, _, w_dt = dtnet(tf.stop_gradient(depth_dn, name='depth_dn_stopped') if config.dnstop else depth_dn, color)
        else:
            depth_dt = normals_dt = None

        with tf.variable_scope("compute_normals") as scp:
            if depth_dn is not None:
                normals_dn = ops.compute_normals(depth_dn, intr, config, conv=False)  # not necessary, to visualize.
                params["depth_dn"] = depth_dn
                params["normals_dn"] = normals_dn
            if depth_dt is not None:  # necessary for shading loss
                normals_dt = ops.compute_normals(depth_dt, intr, config, conv=False)
                params["depth_dt"] = depth_dt
                params["normals_dt"] = normals_dt
            # For visualization during training
            normals_ref = ops.compute_normals(depth_ref, intr, config, conv=False)  # set conv=True for smoother normals
            params["normals_ref"] = normals_ref
            normals_raw = ops.compute_normals(depth_raw, intr, config, conv=False)
            params["normals_raw"] = normals_raw

        # Calculate num of total parameters in the network.
        num_params = 0
        for var in tf.trainable_variables():
            print(var.name, var.get_shape())
            num_params += reduce(mul, var.get_shape().as_list(), 1)
        print("Total parameter: {}".format(num_params))

        # Add total loss of dn_net and dt_net
        loss, self.loss1, self.loss2, dbg_ret = model.loss(depth_dn, depth_dt, depth_ref, config,
                                                           normal_dn=normals_dn, normal_dt=normals_dt,
                                                           normal_gt=normals_ref, color=color,
                                                           mask=params["mask"], abd=params["abd"])
        # Visualize irradiance and albedo map.
        if dtnet:
            params["irrad"] = ops.convertNCHW2NHWC(dbg_ret[0], name='irrad_NHWC')
            params["albedo"] = ops.convertNCHW2NHWC(dbg_ret[1], name='albedo_NHWC')

        with tf.variable_scope("Optimizer"):
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, 300, 0.8)
            self.summaries.append(tf.summary.scalar('lr', learning_rate))
            optimizer = tf.train.AdamOptimizer(learning_rate, 0.9)
            grads, vars = zip(*optimizer.compute_gradients(loss))

            if config.grad_clip:
                grad_global_norm = 200.0
                print('clip grad by global norm {} during training.'.format(grad_global_norm))
                grads, _ = tf.clip_by_global_norm(grads, grad_global_norm)
                # grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
            ## capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
            # # show grads_and_vars distribution
            # for i, gradient in enumerate(grads):
            #     if isinstance(gradient, tf.IndexedSlices):
            #         grad_values = gradient.values
            #     else:
            #         grad_values = gradient
            #     var = vars[i]
            #     self.summaries.append(tf.summary.histogram(var.name, var))
            #     self.summaries.append(tf.summary.histogram(var.name + "/gradients", grad_values))
            #     self.summaries.append(tf.summary.scalar(var.name + "/gradient_norm",
            #                                        tf.global_norm([grad_values])))

            # Show the gradient back-propagated from dt_net to dn_net.
            for i, v in enumerate(vars):
                if 'dn_net/cnv_final/weights' in v.name:
                    self.summaries.append(tf.summary.scalar('dn_net/cnv_final/weights_grad', tf.norm(grads[i])))
            print(vars[0].name)
            self.summaries.append(tf.summary.scalar('dn_net/cnv_0/weights_grad', tf.norm(grads[0])))

            train_op = optimizer.apply_gradients(zip(grads, vars), global_step)
            # train_op = optimizer.minimize(loss, global_step)

        self.train_op = train_op
        self.total_loss = loss
        with tf.name_scope("Losses"):
            self.summaries.append(tf.summary.scalar("total_loss", loss))
            for ls in tf.get_collection(LOSSES_COLLECTION):
                ls_name = ls.name.split('/')[0]
                # print('Losses: ' + ls_name)
                self.summaries.append(tf.summary.scalar(ls_name, ls))


    def fit(self):
        is_training = True
        config, params = self.config, self.params

        # start training from previous global_step
        start_step = self.sess.run(params["global_step"])
        if not start_step == 0:
            print("Start training from previous {} steps".format(start_step))

        for step in range(start_step, config.max_steps):
            t1 = time.time()

            # # dbg filter condition.
            # diff = self.sess.run(self.diff)
            # if diff < config.diff_thres: logging.debug(diff)
            # else: logging.debug('diff too large. discard.'+str(diff))

            loss, loss1, loss2, _ = self.sess.run([self.total_loss, self.loss1, self.loss2, self.train_op],
                          feed_dict={params["is_training"]: is_training})
            print('step {}: loss: {:.2f}\t loss1: {:.2f}\t loss2: {:.2f}'.format(step, loss, loss1, loss2))
            t2 = time.time()

            if step % config.summary_every_n_steps == 0:
                summary_feed_dict = {params["is_training"]: is_training}
                self.make_summary(summary_feed_dict, step)

                eta = (t2 - t1) * (config.max_steps - step + 1)
                print("Finished {}/{} steps, ETA:{:.2f} seconds".format(step, config.max_steps, eta))
                utils.flush_stdout()

            if step % config.save_model_steps == 0:
                self.saver.save(self.sess, os.path.join(config.logdir,
                    "{}-{}".format(config.checkpoint_basename.split('/')[-1], step)))

        self.saver.save(self.sess, os.path.join(config.logdir,
            "{}-{}".format(config.checkpoint_basename.split('/')[-1], config.max_steps)))


    def make_summary(self, feed_dict, step):
        summary = self.sess.run(self.all_summaries, feed_dict=feed_dict)
        self.sv.summary_computed(self.sess, summary, step)


    def pseudo_color(self, gray, low=-1.0, up=1.0, normalize=False):
        """
        clamp 1 ch tensor to range [low, 1] and convert to pseudo-color for visualization.
        :param gray: [N, H, W, 1]
        :param low: -1 or 0
        :param normalize: use min/max value in image as lower/upper bound of scaling.
        :return: pseudo-color tensor [N, H, W, 3] in range [0, 1]
        """
        with tf.variable_scope("pseudo_color"):
            # gray = tf.clip_by_value(gray, low, up, name='clipped_gray')
            if normalize:  # in case output batch has pixel out of range(low, up)
                im_list = tf.unstack(gray, self.config.batch_size)
                im_norm_list = []
                for im in im_list:
                    im_min = tf.reduce_min(im)
                    im_max = tf.reduce_max(im)
                    im_norm = (up-low)*(im-im_min)/(im_max-im_min) + low
                    im_norm_list.append(im_norm)
                gray = tf.stack(im_norm_list, name='gray_norm')
                del im_list, im_norm_list
            else:
                pass

            th_2 = (low + up)   / 2.0
            th_1 = (low + th_2) / 2.0
            th_3 = (th_2 + up)  / 2.0
            mask_1 = tf.less(gray, th_1)
            mask_2 = tf.less(gray, th_2)
            mask_3 = tf.less(gray, th_3)
            zero_const = tf.zeros_like(gray)
            one_const = tf.ones_like(gray)

            red_ch = tf.where(mask_2, zero_const, one_const)
            mask = tf.logical_and(tf.logical_not(mask_2), mask_3, name='mid_right')
            red_ch = tf.where(mask, (gray-th_2)/(th_3-th_2), red_ch)
            green_ch = tf.where(mask_1, (gray-low)/(th_1-low), one_const)
            green_ch = tf.where(mask_3, green_ch, one_const-(gray-th_3)/(up-th_3))
            blue_ch = tf.where(mask_2, one_const, zero_const)
            mask = tf.logical_and(tf.logical_not(mask_1), mask_2, name='mid_left')
            blue_ch = tf.where(mask, one_const-(gray-th_1)/(th_2-th_1), blue_ch)

            color = tf.concat([red_ch, green_ch, blue_ch], axis=3)
            color = tf.clip_by_value(color, 0.0, 1.0, name='pseudo_color')
            return color
