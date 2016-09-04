from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class GAN(object):
  
  def __init__(self, image_size=28, image_dims=1, z_dims=16, batch_size=256):
    self.image_size = image_size
    self.image_dims = image_dims 
    self.z_dims = z_dims
    self.batch_size = batch_size

    self.initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)

    self.generator_dims = [128, 64, 32, 16, image_dims]
    self.generator_shapes = [2, 4, 8, 16, 28]
    self.generator_kernel_sizes = [2, 3, 3, 3, 3]

    self.discriminator_dims = [[5]]
    self.discriminator_kernel_sizes = [[28]]

    weights = self._initialize_weights()
    self.all_weights = weights
    self.real_images = tf.placeholder("float", [self.batch_size, self.image_size, self.image_size, self.image_dims])
    self.fake_images = tf.placeholder("float", [self.batch_size, self.image_size, self.image_size, self.image_dims])

    self.gen_cost = self.get_gen_cost()
    self.disc_cost = dict()
    for disc_i in xrange(len(self.discriminator_dims)):
      self.disc_cost[disc_i] = self.get_disc_cost(disc_i)

    init = tf.initialize_all_variables()
    self.sess = tf.Session()
    self.sess.run(init)


  def _initialize_weights(self):
    all_weights = dict()

    # init generator weights
    prev_layer_dims = self.z_dims
    for layer_i in xrange(len(self.generator_dims)):
      name = 'gen_w' + str(layer_i)
      all_weights[name] = tf.get_variable(name, 
            [self.generator_kernel_sizes[layer_i], 
             self.generator_kernel_sizes[layer_i], 
             self.generator_dims[layer_i], 
             prev_layer_dims], 
          tf.float32,
          self.initializer)
      name = 'gen_b' + str(layer_i)
      all_weights[name] = tf.get_variable(name, 
          [self.generator_dims[layer_i]], tf.float32, tf.constant_initializer(0.0))
      prev_layer_dims = self.generator_dims[layer_i]

    # init discriminator weights
    for disc_i in xrange(len(self.discriminator_dims)):
      prev_layer_dims = self.image_dims
      for layer_i in xrange(len(self.discriminator_dims[disc_i])):
        name = 'disc' + str(disc_i) + '_w' + str(layer_i)
        all_weights[name] = tf.get_variable(name, 
              [self.discriminator_kernel_sizes[disc_i][layer_i], 
               self.discriminator_kernel_sizes[disc_i][layer_i], 
               prev_layer_dims,
               self.discriminator_dims[disc_i][layer_i]],
            tf.float32,
            self.initializer)

        name = 'disc' + str(disc_i) + '_b' + str(layer_i)
        all_weights[name] = tf.get_variable(name, [self.discriminator_dims[disc_i][layer_i]], tf.float32, tf.constant_initializer(0.0))
        prev_layer_dims = self.discriminator_dims[disc_i][layer_i]

    return all_weights

  def generator(self, z):
    #z = tf.placeholder("float", [self.batch_size, 1, 1, self.z_dims])
    prev_layer = z
    for layer_i in xrange(len(self.generator_dims)):
      conv = tf.nn.conv2d_transpose(prev_layer, self.all_weights['gen_w'+str(layer_i)], 
          output_shape=[self.batch_size, 
                self.generator_shapes[layer_i], 
                self.generator_shapes[layer_i], 
                self.generator_dims[layer_i]], 
          strides=[1, 2, 2, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, self.all_weights['gen_b'+str(layer_i)])
      prev_layer = tf.nn.relu(bias, name=scope.name)
      
    return prev_layer


  def discriminator_logits(self, disc_i, images):
    prev_layer = images
    for layer_i in xrange(len(self.discriminator_dims[disc_i])):
      conv = tf.nn.conv2d(prev_layer, 
          self.all_weights['disc'+str(disc_i)+'_w'+str(layer_i)],
          strides=[1, 2, 2, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, self.all_weights['disc'+str(disc_i)+'_b'+str(layer_i)])
      prev_layer = tf.nn.relu(bias, name=scope.name)
    return prev_layer

  def get_gen_cost(self):
    pass

  def get_disc_cost(self, i):
    pass

  def train_step(self):
    pass


