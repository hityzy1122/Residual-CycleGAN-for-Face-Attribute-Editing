#tfrecords
import tensorflow as tf
import random
import os
from os import scandir
"""
程序功能：将jpg格式的文件转换成tfrecords格式文件，减少内存查找浪费的时间
输入：
--X_input_dir 存有大量jpg图片的文件夹
--X_output_file 输出的/path/to/*.tfrecords文件的路径
"""
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('X_input_dir', 'E:\\Zhiyang\\project\\datasets\\Attribute\\sun-eyeglass',
                       'X input directory, default: ')

tf.flags.DEFINE_string('X_output_file', 'E:\\Zhiyang\\project\\datasets\\Attribute\\tfrecords\\sun_eyeglass.tfrecords',
                       'X output tfrecords file, default: ')


def data_reader(input_dir, shuffle=True):
  """从 input_dir 读取文件并打乱顺序
  输入:
    input_dir: "/path/to/dir"
  返回:
    文件夹下打乱顺序的文件名
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths



def _bytes_feature(value):
  """向实例中输入二进制特征."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """将读到的图片数据转化成tfrecords一个实例
  输入:
    file_path: '/path/to/example.JPG'
    image_buffer: 二进制图片
  返回:
    一个实例
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example


def data_writer(input_dir, output_file):
  """图片数据写入tfrecords
  """
  file_paths = data_reader(input_dir)

  # 建立一个存储tfrecords的文件夹
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error as e:
    pass

  images_num = len(file_paths)

  # 建立一个writer
  writer = tf.python_io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    with tf.gfile.FastGFile(file_path, 'rb') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()

def main(unused_argv):
  print("Convert X data to tfrecords...")
  data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)


if __name__ == '__main__':
  tf.app.run()
