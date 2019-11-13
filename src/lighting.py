import tensorflow as tf


def H0(normals):
    B, C, L = normals.shape.as_list()
    # return tf.ones([B, 1, L], dtype=tf.float32)
    return tf.ones_like(normals, dtype=tf.float32)[:, 0:1, :]


def H1(normals):
    return normals[:, 1:2, :]


def H2(normals):
    return normals[:, 2:3, :]


def H3(normals):
    return normals[:, 0:1, :]


def H4(normals):
    return tf.multiply(normals[:, 0:1, :], normals[:, 1:2, :])


def H5(normals):
    return tf.multiply(normals[:, 1:2, :], normals[:, 2:3, :])


def H6(normals):
    return - tf.multiply(normals[:, 0:1, :], normals[:, 0:1, :]) \
           - tf.multiply(normals[:, 1:2, :], normals[:, 1:2, :]) \
           + 2 * tf.multiply(normals[:, 2:3, :], normals[:, 2:3, :])


def H7(normals):
    return tf.multiply(normals[:, 2:3, :], normals[:, 0:1, :])


def H8(normals):
    return tf.multiply(normals[:, 0:1, :], normals[:, 0:1, :]) \
           - tf.multiply(normals[:, 1:2, :], normals[:, 1:2, :])


def get_H(normals, mask):
    """
    :param normals: (B,3,H,W)
    :param mask: (B,1,H,W)
    :return: (B,9,HW)
    """
    B, C, H, W = normals.shape.as_list()
    normals = tf.reshape(normals, [-1, C, H*W])
    mask = tf.reshape(mask, [-1, 1, H*W])
    return tf.multiply(tf.concat([H0(normals), H1(normals), H2(normals), H3(normals), H4(normals),
                                  H5(normals), H6(normals), H7(normals), H8(normals)], 1), mask)


def get_lighting(normals, image, abd, mask, rm_graz=False, eps=1e-4):
    """    
    :param normals: BCHW
    :param image: BCH1
    :param abd: BCHW
    :param mask: BCHW
    :return: lighting with shape (B,9), LH with shape (B,1,H,W)
    """

    ## Remove normals at high grazing angle
    if rm_graz:
        mask_angle = tf.cast(tf.greater(normals[:, 2:3, :, :], 0.5), tf.float32)
        mask = tf.multiply(mask, mask_angle)

    image = (tf.multiply(image, mask) + 1.0) / 2  # transform to [0,1] for lighting estimation
    B, C, H, W = image.shape.as_list()
    image = tf.reshape(image, [-1, C, H*W])
    image = tf.transpose(image, perm=[0, 2, 1])
    A = get_H(normals, mask)

    # Use offline estimated albedo
    if abd is not None:
        abd = tf.reshape(abd, [1, -1, H*W])
        A = tf.multiply(A, abd)
    A_t = tf.transpose(A, perm=[0, 2, 1])

    AA_t = tf.matmul(A, A_t) + eps*tf.eye(9, name='lighting_inverse_eps')
    # TODO: image rescale to [0,1]?
    lighting = tf.squeeze(tf.matmul(tf.matmul(tf.matrix_inverse(AA_t), A), image), axis=2)
    LH = tf.reshape(tf.matmul(tf.expand_dims(lighting, 1), A), [-1, 1, H, W])
    return lighting, LH, mask


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    print("Test for lighting")
    normals = Image.open('../dataset/20170907/group1/high_quality_depth_n/frame_000001_n.png').convert('RGB')
    image = Image.open('../dataset/20170907/group1/color_map/frame_000001.png').convert('L')
    mask = Image.open('../dataset/20170907/group1/mask/frame_000001.png').convert('L')
    normals = tf.convert_to_tensor(np.asarray(normals), dtype=tf.float32) / 255.0
    image = tf.expand_dims(tf.convert_to_tensor(np.asarray(image), dtype=tf.float32), 0) / 255.0
    mask = tf.expand_dims(tf.convert_to_tensor(np.asarray(mask), dtype=tf.float32), 0) / 255.0
    normals = tf.expand_dims(tf.transpose(normals, perm=[2, 0, 1]), 0)
    image = tf.expand_dims(image, 0)
    mask = tf.expand_dims(mask, 0)
    print(normals.shape, image.shape, mask.shape)
    sess = tf.Session()
    lighting, _ = sess.run(get_lighting(normals, image, None, mask))
    print(lighting)
