#utils

import tensorflow as tf
import random


def convert2float(image):
  """
  将0~255转换到-1~1
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0


class ImagePool:
  """
  在线样本池：
  在线存储50个样本，有50%概率选择样本池中样本训练，50%概率选择当前样本训练
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image
