import os
import argparse
import logging
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
slim = tf.contrib.slim

import trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--checkpoint_basename", type=str)
    parser.add_argument("--index_file", type=str, default="train/train_albedo.csv")
    parser.add_argument("--has_mask", action='store_true')
    parser.add_argument("--has_abd", action='store_true')

    parser.add_argument("--dnnet", type=str)
    parser.add_argument("--dtnet", type=str)
    parser.add_argument("--dnstop", action='store_true')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--aux_type", type=str, default="PNG")
    parser.add_argument("--diff_thres", type=float, default=2.0)  # 2.0 for k2, 3.0 for k1.
    parser.add_argument("--low_thres", type=float, default=500.0)
    parser.add_argument("--up_thres", type=float, default=3000.0)

    parser.add_argument("--rand_crop", action='store_true')
    parser.add_argument("--rand_flip", action='store_true')
    parser.add_argument("--rand_scale", action='store_true')
    parser.add_argument("--rand_depth_shift", action='store_true')
    parser.add_argument("--rand_brightness", action='store_true')

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--grad_clip", action='store_true')
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.0001)

    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--save_model_steps", type=int, default=5000)
    parser.add_argument("--min_after_dequeue", type=int, default=128)
    parser.add_argument("--use_shuffle_batch", type=bool, default=True)

    parser.add_argument("--max_to_keep", type=int, default=100)
    parser.add_argument("--summary_every_n_steps", type=int, default=20)

    return parser.parse_args()


def main(args):
    logging.basicConfig(
      level=logging.DEBUG,
      format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
      filename=os.path.join(args.logdir, 'logging.txt'))
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    filename = os.path.realpath(args.index_file)
    if not os.path.isfile(filename):
        raise ValueError('No such index_file: {}'.format(filename))
    else:
        print("Reading csv file: {}".format(filename))

    with open(filename, "r") as f:
        line = f.readline().strip()
        input_path = line.split(',')[0]
        if not os.path.exists(input_path):
            raise ValueError('Input path in csv not exist: {}'.format(input_path))

    t = trainer.Trainer(filename, args)
    t.fit()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
