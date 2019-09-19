#train
#! /home/yuzhiyang/anaconda3/bin/python
import tensorflow as tf
from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool
import ops
a=tf.layers.conv2d()
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')

tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')

tf.flags.DEFINE_integer('channels', 3, 'image channels, default: 3')

# 使用最小二乘GAN
tf.flags.DEFINE_bool('use_lsgan', True, 'use lsgan (mean squared error) or cross entropy loss, default: True')
# 使用 instance norm
tf.flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm, default: instance')
# cycle-loss 的系数
tf.flags.DEFINE_integer('lambda1', 15.0, 'weight for forward cycle loss (X->Y->X), default: 15.0')
tf.flags.DEFINE_integer('lambda2', 15.0, 'weight for backward cycle loss (Y->X->Y), default: 15.0')
# L1(residual) 的系数
tf.flags.DEFINE_integer('lambda3', 1e-5, 'weight for l1 loss, default: 1e-5')
# TV(residual) 的系数
tf.flags.DEFINE_integer('lambda4', 1e-6, 'weight for TV loss, default: 1e-5')
# IP-loss 的系数
tf.flags.DEFINE_integer('lambda5', 1e-4, 'weight for IP loss, default: 1e-4')
# adam 优化方法的参数
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
# 在线样本池的大小
tf.flags.DEFINE_float('pool_size', 50, 'size of pool, default: 50')
# G网络第一层卷积核数量
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
# tfrecord 路径
tf.flags.DEFINE_string('X', 'data/tfrecords/no_glass.tfrecords', 'X tfrecords file for training')
tf.flags.DEFINE_string('Y', 'data/tfrecords/with_glass.tfrecords', 'Y tfrecords file for training')
# 存储中间训练结果的路径
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
# 存储中间验证结果的路径
tf.flags.DEFINE_string('res_im_path', './img_output', 'folder of validation result')
# 总共迭代次数
tf.flags.DEFINE_integer('epho', 75000, 'number of epho')

def train():

  # 如果存储中间训练结果的路径设置不为None 就从路径中读取数据继续训练，如果为None则建立一个新的，以时间命名的文件夹存储训练结果
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
      os.makedirs(FLAGS.res_im_path)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    # 初始化 cyclegan 类
    cycle_gan = CycleGAN(FLAGS)

    # 构建图
    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, real_y, real_x = cycle_gan.model()
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    # 初始化summary
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver(max_to_keep=10)

  with tf.Session(graph=graph) as sess:
    # 如果存储中间训练结果的路径设置不为None 就从路径中读取数据继续训练
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    # 初始化样本队列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      # 初始化在线样本池
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():
        # get previously generated images
        fake_y_val, fake_x_val, real_y_in, real_x_in = sess.run([fake_y, fake_x, real_y, real_x])

        # train
        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
              sess.run(
                  [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                             cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
              )
        )

        train_writer.add_summary(summary, step)
        train_writer.flush()
        # 输出当前状态
        if step % 1 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss   : {}'.format(G_loss_val))
          logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
          logging.info('  F_loss   : {}'.format(F_loss_val))
          logging.info('  D_X_loss : {}'.format(D_X_loss_val))

        if step % 1000 == 0:
            ops.save_img_result(fake_y_val, fake_x_val, real_y_in, real_x_in, FLAGS.res_im_path, step)

        if step % 1000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)

        step += 1
        if step == FLAGS.epho:
            coord.request_stop()  # 发出停止训练信号

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      ops.save_img_result(fake_y_val, fake_x_val, real_y_in, real_x_in, FLAGS.res_im_path, step)
      logging.info("Model saved in file: %s" % save_path)

      coord.request_stop()  # 停止训练
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()