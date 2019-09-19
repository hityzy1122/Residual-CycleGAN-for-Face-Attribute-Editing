# model
import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
import numpy as np
import cv2
REAL_LABEL = 0.9

class CycleGAN:
  def __init__(self, FLAGS):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer,  循环一致性损失(X->Y->X)
      lambda2: integer,  循环一致性损失(Y->X->Y)
      use_lsgan: boolean
      norm: instance
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambda1 = FLAGS.lambda1
    self.lambda2 = FLAGS.lambda2
    self.lambda3 = FLAGS.lambda3
    self.lambda4 = FLAGS.lambda4
    self.lambda5 = FLAGS.lambda5

    self.use_lsgan = FLAGS.use_lsgan
    use_sigmoid = not FLAGS.use_lsgan
    self.batch_size = FLAGS.batch_size
    self.image_size = FLAGS.image_size
    self.channels = FLAGS.channels
    self.learning_rate = FLAGS.learning_rate
    self.beta1 = FLAGS.beta1
    self.X_train_file = FLAGS.X
    self.Y_train_file = FLAGS.Y

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=FLAGS.norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training, norm=FLAGS.norm, image_size=FLAGS.image_size)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=FLAGS.norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.channels])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.channels])
    self.mask = None
    self.load_mask()

  def load_mask(self):
      mask = np.float32(cv2.imread('mask.jpg', int(-self.channels / 2.0 + 0.5)))
      mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
      if self.channels == 1:
          mask = mask[:, :, np.newaxis]
      mask[mask < 255.0] = 255.0
      mask /= 255.0
      mask = mask[np.newaxis, ...]
      self.mask = tf.constant(np.repeat(mask, self.batch_size, axis=0), dtype=tf.float32)


  def model(self):
    # 从tfrecords 建立一个输入队列
    X_reader = Reader(self.X_train_file, name='X', image_size=self.image_size, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y', image_size=self.image_size, batch_size=self.batch_size)

    # 从队列中弹出一个样本
    x = X_reader.feed()
    y = Y_reader.feed()
    # ------------------------------------------------------------------------------------
    real_x = x
    real_y = y

    # 计算循环一致性损失
    cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

    # X -> Y 残差结构---------------------------------------------------------------------------------
    fake_y_residual = self.G(x)
    fake_y = ops.residual2img(residual=fake_y_residual, input=x, mask=self.mask, name='fake_y')
    ip_y = ops.ip_loss(fake_y, x)
    # -----------------------------------------------------------------------------------------------
    G_gan_loss = self.generator_loss(self.D_Y, fake_y)
    G_loss =  G_gan_loss + cycle_loss + \
              self.lambda3*tf.reduce_mean(tf.abs(fake_y_residual)) + \
              self.lambda4*tf.reduce_mean(tf.image.total_variation(fake_y_residual)) + \
              self.lambda5*ip_y

    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y)

    # Y -> X 残差结构---------------------------------------------------------------------------------
    fake_x_residual = self.F(y)
    fake_x = ops.residual2img(residual=fake_x_residual, input=y, mask=self.mask, name='fake_y')
    ip_x = ops.ip_loss(fake_x, y)
    # -----------------------------------------------------------------------------------------------
    F_gan_loss = self.generator_loss(self.D_X, fake_x)
    F_loss = F_gan_loss + cycle_loss + \
             self.lambda3*tf.reduce_mean(tf.abs(fake_x_residual)) + \
             self.lambda4*tf.reduce_mean(tf.image.total_variation(fake_x_residual)) + \
             self.lambda5*ip_x

    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x)

    # summary
    # tf.summary.histogram('D_Y/true', self.D_Y(y))
    # tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
    # tf.summary.histogram('D_X/true', self.D_X(x))
    # tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)

    # tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
    # tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
    # tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
    # tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))

    return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x, real_y, real_x

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def discriminator_loss(self, D, y, fake_y):
    """
    LSGAN判别网络的损失函数
    """
    error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
    error_fake = tf.reduce_mean(tf.square(D(fake_y)))

    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y):
    """
    LSGAN生成网络的损失函数
    """
    loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    return loss

  def cycle_consistency_loss(self, G, F, x, y):
    """ 循环一致性损失
    """
    residual_x_y = G(x)
    fake_x_y = ops.residual2img(residual=residual_x_y, input=x, mask=self.mask, name='fake_x_y')
    ip_x_y = ops.ip_loss(fake_x_y, x, if_resuse=False)

    residual_x_y_x = F(fake_x_y)
    fake_x_y_x = ops.residual2img(residual=residual_x_y_x, input=fake_x_y, mask=self.mask, name='fake_x_y_x')
    ip_x_y_x = ops.ip_loss(fake_x_y_x, fake_x_y, if_resuse=True)

    residual_y_x = F(y)
    fake_y_x = ops.residual2img(residual=residual_y_x, input=y, mask=self.mask, name='fake_y_x')
    ip_y_x = ops.ip_loss(fake_y_x, y, if_resuse=True)

    residual_y_x_y = G(fake_y_x)
    fake_y_x_y = ops.residual2img(residual=residual_y_x_y, input=fake_y_x, mask=self.mask, name='fake_y_x_y' )
    ip_y_x_y = ops.ip_loss(fake_y_x_y, fake_y_x,if_resuse=True)

    forward_loss = tf.reduce_mean(tf.abs(fake_x_y_x - x))
    backward_loss = tf.reduce_mean(tf.abs(fake_y_x_y - y))

    residual_l1_loss = tf.reduce_mean(tf.abs(residual_x_y)) + \
                       tf.reduce_mean(tf.abs(residual_x_y_x)) + \
                       tf.reduce_mean(tf.abs(residual_y_x)) + \
                       tf.reduce_mean(tf.abs(residual_y_x_y))

    TV_loss = tf.reduce_mean(tf.image.total_variation(residual_x_y)) + \
              tf.reduce_mean(tf.image.total_variation(residual_x_y_x)) + \
              tf.reduce_mean(tf.image.total_variation(residual_y_x)) + \
              tf.reduce_mean(tf.image.total_variation(residual_y_x_y))

    IP_loss = (ip_x_y + ip_x_y_x + ip_y_x + ip_y_x_y)/4

    loss = self.lambda1*forward_loss + self.lambda2*backward_loss + \
           self.lambda3*residual_l1_loss + self.lambda4*TV_loss + self.lambda5*IP_loss
    return loss