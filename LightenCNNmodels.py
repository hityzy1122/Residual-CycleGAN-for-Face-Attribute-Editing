#lightencnn
import tensorflow as tf
import numpy as np
import pickle
# LightCNN
with open('LightenNet_C.caffeparam','rb') as f:
    weights, biases = pickle.load(f)


def LightCNN_C(input, is_train=False, reuse=False):
    """
    用于产生IP-loss的lightencnn
    参考网站：https://github.com/AlfredXiangWu/face_verification_experiment
    """
    end_points = {}
    with tf.variable_scope('LightCNN') as scope:
        if reuse:
            scope.reuse_variables()
        # Layer1---------------------------------------------------------------------------------------------------
        weight = weights['conv1']
        bias = biases['conv1']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])

        conv1 = tf.layers.conv2d(inputs=input,
                                 filters=weight.shape[3],
                                 kernel_size=[weight.shape[0], weight.shape[1]],
                                 strides=[1, 1],
                                 padding="same",
                                 kernel_initializer=tf.constant_initializer(weight),
                                 bias_initializer=tf.constant_initializer(bias),
                                 trainable=is_train,
                                 name='conv1')
        end_points['conv1'] = conv1

        # max out activate
        slice1_1, slice1_2 = tf.split(value=conv1,
                                      num_or_size_splits=2,
                                      axis=3,
                                      name='slice1')

        eltwise1 = tf.maximum(slice1_1, slice1_2, name='eltwise1')
        end_points['eltwise1'] = eltwise1
        # max pooling
        pool1 = tf.layers.max_pooling2d(inputs=eltwise1,
                                        pool_size=[2, 2],
                                        strides=[2, 2],
                                        padding='valid',
                                        name='pool1')
        end_points['pool1'] = pool1

        # Layer2-------------------------------------------------------------------------------
        weight = weights['conv2a']
        bias = biases['conv2a']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv2a = tf.layers.conv2d(inputs=pool1,
                                  filters=weight.shape[3],
                                  kernel_size=[weight.shape[0], weight.shape[1]],
                                  strides=[1, 1],
                                  padding="same",
                                  kernel_initializer=tf.constant_initializer(weight),
                                  bias_initializer=tf.constant_initializer(bias),
                                  trainable=is_train,
                                  name='conv2a')
        end_points['conv2a'] = conv2a
        slice2a_1, slice2a_2 = tf.split(value=conv2a,
                                        num_or_size_splits=2,
                                        axis=3,
                                        name='slice2a')

        eltwise2a = tf.maximum(slice2a_1, slice2a_2, name='eltwise2a')
        end_points['eltwise2a'] = eltwise2a

        weight = weights['conv2']
        bias = biases['conv2']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv2 = tf.layers.conv2d(inputs=eltwise2a,
                                 filters=weight.shape[3],
                                 kernel_size=[weight.shape[0], weight.shape[1]],
                                 strides=[1, 1],
                                 padding="same",
                                 kernel_initializer=tf.constant_initializer(weight),
                                 bias_initializer=tf.constant_initializer(bias),
                                 trainable=is_train,
                                 name='conv2')
        end_points['conv2'] = conv2
        slice2_1, slice2_2 = tf.split(value=conv2,
                                      num_or_size_splits=2,
                                      axis=3,
                                      name='slice2')

        eltwise2 = tf.maximum(slice2_1, slice2_2, name='eltwise2')
        end_points['eltwise2'] = eltwise2

        pool2 = tf.layers.max_pooling2d(inputs=eltwise2,
                                        pool_size=[2, 2],
                                        strides=[2, 2],
                                        padding='valid',
                                        name='pool2')
        end_points['pool2'] = pool2
        # Layer3-------------------------------------------------------------------------------
        weight = weights['conv3a']
        bias = biases['conv3a']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv3a = tf.layers.conv2d(inputs=pool2,
                                  filters=weight.shape[3],
                                  kernel_size=[weight.shape[0], weight.shape[1]],
                                  strides=[1, 1],
                                  padding="same",
                                  kernel_initializer=tf.constant_initializer(weight),
                                  bias_initializer=tf.constant_initializer(bias),
                                  trainable=is_train,
                                  name='conv3a')

        end_points['conv3a'] = conv3a

        slice3a_1, slice3a_2 = tf.split(value=conv3a,
                                        num_or_size_splits=2,
                                        axis=3,
                                        name='slice3a')

        eltwise3a = tf.maximum(slice3a_1, slice3a_2, name='eltwise3a')

        end_points['eltwise3a'] = eltwise3a

        weight = weights['conv3']
        bias = biases['conv3']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv3 = tf.layers.conv2d(inputs=eltwise3a,
                                 filters=weight.shape[3],
                                 kernel_size=[weight.shape[0], weight.shape[1]],
                                 strides=[1, 1],
                                 padding="same",
                                 kernel_initializer=tf.constant_initializer(weight),
                                 bias_initializer=tf.constant_initializer(bias),
                                 trainable=is_train,
                                 name='conv3')

        end_points['conv3'] = conv3

        slice3_1, slice3_2 = tf.split(value=conv3,
                                      num_or_size_splits=2,
                                      axis=3,
                                      name='slice3')

        eltwise3 = tf.maximum(slice3_1, slice3_2, name='eltwise3')
        end_points['eltwise3'] = eltwise3

        pool3 = tf.layers.max_pooling2d(inputs=eltwise3,
                                        pool_size=[2, 2],
                                        strides=[2, 2],
                                        padding='valid',
                                        name='pool3')
        end_points['pool3'] = pool3
        # Layer4-------------------------------------------------------------------------------
        weight = weights['conv4a']
        bias = biases['conv4a']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv4a = tf.layers.conv2d(inputs=pool3,
                                  filters=weight.shape[3],
                                  kernel_size=[weight.shape[0], weight.shape[1]],
                                  strides=[1, 1],
                                  padding="same",
                                  kernel_initializer=tf.constant_initializer(weight),
                                  bias_initializer=tf.constant_initializer(bias),
                                  trainable=is_train,
                                  name='conv4a')

        end_points['conv4a'] = conv4a
        slice4a_1, slice4a_2 = tf.split(value=conv4a,
                                        num_or_size_splits=2,
                                        axis=3,
                                        name='slice4a')

        eltwise4a = tf.maximum(slice4a_1, slice4a_2, name='eltwise4a')

        end_points['eltwise4a'] = eltwise4a

        weight = weights['conv4']
        bias = biases['conv4']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv4 = tf.layers.conv2d(inputs=eltwise4a,
                                 filters=weight.shape[3],
                                 kernel_size=[weight.shape[0], weight.shape[1]],
                                 strides=[1, 1],
                                 padding="same",
                                 kernel_initializer=tf.constant_initializer(weight),
                                 bias_initializer=tf.constant_initializer(bias),
                                 trainable=is_train,
                                 name='conv4')

        end_points['conv4'] = conv4
        slice4_1, slice4_2 = tf.split(value=conv4,
                                      num_or_size_splits=2,
                                      axis=3,
                                      name='slice4')

        eltwise4 = tf.maximum(slice4_1, slice4_2, name='eltwise4')
        end_points['eltwise4'] = eltwise4
        # Layer5-------------------------------------------------------------------------------
        weight = weights['conv5a']
        bias = biases['conv5a']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv5a = tf.layers.conv2d(inputs=eltwise4,
                                  filters=weight.shape[3],
                                  kernel_size=[weight.shape[0], weight.shape[1]],
                                  strides=[1, 1],
                                  padding="same",
                                  kernel_initializer=tf.constant_initializer(weight),
                                  bias_initializer=tf.constant_initializer(bias),
                                  trainable=is_train,
                                  name='conv5a')
        end_points['conv5a'] = conv5a
        slice5a_1, slice5a_2 = tf.split(value=conv5a,
                                        num_or_size_splits=2,
                                        axis=3,
                                        name='slice5a')

        eltwise5a = tf.maximum(slice5a_1, slice5a_2, name='eltwise5a')

        end_points['eltwise5a'] = eltwise5a

        weight = weights['conv5']
        bias = biases['conv5']
        weight = np.transpose(weight, axes=[2, 3, 1, 0])
        conv5 = tf.layers.conv2d(inputs=eltwise5a,
                                 filters=weight.shape[3],
                                 kernel_size=[weight.shape[0], weight.shape[1]],
                                 strides=[1, 1],
                                 padding="same",
                                 kernel_initializer=tf.constant_initializer(weight),
                                 bias_initializer=tf.constant_initializer(bias),
                                 trainable=is_train,
                                 name='conv5')

        end_points['conv5'] = conv5

        slice5_1, slice5_2 = tf.split(value=conv5,
                                      num_or_size_splits=2,
                                      axis=3,
                                      name='slice5')

        eltwise5 = tf.maximum(slice5_1, slice5_2, name='eltwise5')

        end_points['eltwise5'] = eltwise5

        pool4 = tf.layers.max_pooling2d(inputs=eltwise5,
                                        pool_size=[2, 2],
                                        strides=[2, 2],
                                        padding='valid',
                                        name='pool4')
        end_points['pool4'] = pool4
        return end_points



if __name__ == '__main__':
    input = tf.placeholder(tf.float32, shape=[1, 128, 128, 1], name='LC_input')
    end_points = LightCNN_C(input, is_train=False, reuse=False)
