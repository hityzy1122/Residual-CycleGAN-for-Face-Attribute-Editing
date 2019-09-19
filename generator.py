#generator
import tensorflow as tf
import ops
import utils
import numpy as np
import cv2

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    """
    输入:
      input: batch_size x width x height x 3
    返回:
      output: 输入输出分辨率一致
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
      d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

      if self.image_size <= 128:
        # use 6 residual blocks for 128x128 images
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128)
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)

      # fractional-strided convolution
      u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
      u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

      output = ops.c7s1_k(u32, 3, norm=None,
          activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output  # change for residual cycle gan

# --------------------------------------------------------------------------------------------------
  def res_add_input(self, input):
    """
    测试时使用，使用时直接从G网络获取输出
    """
    mask = np.float32(cv2.imread('mask.jpg', cv2.IMREAD_UNCHANGED))
    mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
    mask[mask <= 25.0] = 25.0
    mask /= 255.0
    mask = mask[np.newaxis, ...]
    mask = tf.constant(np.repeat(mask, 1, axis=0), dtype=tf.float32)

    residual = (self.__call__(input)) * mask
    image = tf.clip_by_value((residual + input), clip_value_min=-1, clip_value_max=1)
    return image, residual