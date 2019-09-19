#opts
import tensorflow as tf
import cv2
import numpy as np
import LightenCNNmodels


def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """7X7 卷积+norm+relu 卷积核个数为k,步长为1
  参数:
    input: 4D tensor
    k: integer, 卷积核个数 (output depth)
    norm: 'instance'
    activation: 'relu'
    name: string, e.g. 'c7sk-32'
    is_training: True or False
    name: string
    reuse: True or False
  返回:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
  """ 降采样卷积，卷积核为3X3+norm+relu, 卷积核个数为k,步长为2
  参数:
    input: 4D tensor
    k: integer, 卷积核个数 (output depth)
    norm: 'instance' or 'batch' or None
    is_training: True or False
    name: string
    reuse: True or False
    name: string
  返回值:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(conv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output

def Rk(input, k,  reuse=False, norm='instance', is_training=True, name=None):
  """
  残差块，卷积核个数为3X3, 输入输出的维度一致
  参数:
    input: 4D Tensor
    k: integer, 卷积核数量
    reuse: True or False
    name: string
  返回值:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights1 = _weights("weights1",
        shape=[3, 3, input.get_shape()[3], k])
      padded1 = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _norm(conv1, is_training, norm)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",
        shape=[3, 3, relu1.get_shape()[3], k])

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _norm(conv2, is_training, norm)
    output = input+normalized2
    return output

def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6):

  """
     堆叠残差块
  """
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output

def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
  """ 反卷积（分数步长卷积）：卷积核3X3+norm+relu, 卷积核个数k,步长1/2
  参数:
    input: 4D tensor
    k: integer, 卷积核个数
    norm: 'instance'
    is_training: True or False
    reuse: True or False
    name: string
    output_size: integer,输出图的分辨率
  返回值:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",
      shape=[3, 3, k, input_shape[3]])

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, k]
    fsconv = tf.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(fsconv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output


# Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
  """ 4X4 卷积+norm+leakyrelu,卷积核个数为k，步长为2
  参数:
    input: 4D tensor
    k: integer, 卷积核个数
    slope: LeakyReLU斜率
    stride: integer
    norm: 'instance'
    is_training: True or False
    reuse: True or False
    name: string, e.g. 'C64'
  返回:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_training, norm)
    output = _leaky_relu(normalized, slope)
    return output

def last_conv(input, reuse=False, use_sigmoid=False, name=None):

  """
  D网络最后一层卷积(1 filter with size 4x4, stride 1)
  参数:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: False 在lsgan中取消激活函数
    name: string
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases", [1])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      output = tf.sigmoid(output)
    return output

### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
  """ 初始化权重
  输入:
    name: name of the variable
    shape: list of ints
    mean: Gaussian均值
    stddev: Gaussian 方差
  返回:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _biases(name, shape, constant=0.0):
  """
  偏置常数初始化
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_training, norm='instance'):
  """ instance-norm or batch-norm
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope("batch_norm"):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)

def save_img_result(fake_y_val, fake_x_val, real_y_in, real_x_in, path, epho):
  """
   存储中间训练结果
   输入两组样本和相应的输出
  """
  res_img = np.concatenate((real_x_in[0], fake_y_val[0], real_y_in[0], fake_x_val[0]), axis=1)
  res_img = (res_img+1)*128
  res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
  cv2.imwrite(path+'/'+str(epho)+'.jpg', res_img)

def residual2img(residual, input, mask, name=None):
  """
  将残差图片转化为生成图片
   输入：残差和输入
   输出：clip(残差+输入)
  """
  fake_add = residual*mask + input
  fake = tf.clip_by_value(fake_add, clip_value_min=-1, clip_value_max=1, name=name)
  return fake

def ip_loss(img_out, img_in, if_resuse=True):
  """
  计算identity preserving loss
   输入： 一组图片
   输出：IP-loss
  """
  out_gray = ((tf.reduce_sum(img_out, axis=3, name='out_gray', keep_dims=True) / 3) + 1) / 2
  in_gray = ((tf.reduce_sum(img_in, axis=3, name='in_gray', keep_dims=True) / 3) + 1) / 2

  light_cnn_out = LightenCNNmodels.LightCNN_C(input=out_gray, is_train=False, reuse=if_resuse)
  light_cnn_in = LightenCNNmodels.LightCNN_C(input=in_gray, is_train=False, reuse=True)

  layer1 = light_cnn_out['pool4'] - light_cnn_in['pool4']
  layer2 = light_cnn_out['eltwise4'] - light_cnn_in['eltwise4']
  ip_loss = 0.5*(tf.reduce_mean(tf.abs(layer1)) + tf.reduce_mean(tf.abs(layer2)))
  return ip_loss