from __future__ import division, print_function, absolute_import

from datetime import datetime
import os.path
import time
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import ops
import flags

class GAN(object):
  
    def __init__(self, image_size=28, image_dim=1, z_dim=16, batch_size=256):
        self.image_size = image_size
        self.image_dim = image_dim 
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)

        self.generator_params = {
            'dim' : [128, 64, 32, 16, image_dim],
            'shape' : [2, 4, 7, 14, 28],
            'ksize' : [2, 4, 4, 4, 4]
            }

        self.discriminators_params = [
                {
                    'dim': [5, 2],
                    'ksize' : [28, 1]
                }
            ]

        self.all_weights, self.batch_norms, self.gen_vars, self.disc_vars  = self._initialize_params()

        self.images = tf.placeholder("float", [self.batch_size, self.image_size, self.image_size, self.image_dim])
        #self.z = tf.placeholder("float", [self.batch_size, 1, 1, self.z_dim])
        self.z = tf.random_normal([self.batch_size, 1, 1, self.z_dim])


        true_labels = np.concatenate(
                [np.ones([self.batch_size, 1]), np.zeros([self.batch_size, 1])], 
                1).astype('float32')
        self.true_labels = tf.convert_to_tensor(true_labels)
        self.false_labels = tf.convert_to_tensor(1-true_labels)


    def _initialize_params(self):
        all_weights = {}
        batch_norms = {}
        gen_vars = []
        disc_vars = []

        # init generator weights
        prev_layer_dim = self.z_dim
        for layer_i in xrange(len(self.generator_params['dim'])):
            name = 'gen_w' + str(layer_i)
            all_weights[name] = ops.variable(name, 
                  [self.generator_params['ksize'][layer_i], 
                   self.generator_params['ksize'][layer_i], 
                   self.generator_params['dim'][layer_i], 
                   prev_layer_dim], 
                self.initializer)
            gen_vars.append(all_weights[name])

            if layer_i+1==len(self.generator_params['dim']):
                name = 'gen_b' + str(layer_i)
                all_weights[name] = ops.variable(name, 
                    [self.generator_params['dim'][layer_i]], 
                    )
                gen_vars.append(all_weights[name])
            else:
                name = 'gen_bn' + str(layer_i)
                batch_norms[name] = ops.batch_norm(self.generator_params['dim'][layer_i], name=name)
            prev_layer_dim = self.generator_params['dim'][layer_i]

        # init discriminator weights
        for disc_i in xrange(len(self.discriminators_params)):
            prev_layer_dim = self.image_dim
            cur_params = self.discriminators_params[disc_i]
            for layer_i in xrange(len(cur_params['dim'])):
                name = 'disc' + str(disc_i) + '_w' + str(layer_i)
                all_weights[name] = ops.variable(name, 
                      [cur_params['ksize'][layer_i], 
                       cur_params['ksize'][layer_i], 
                       prev_layer_dim,
                       cur_params['dim'][layer_i]],
                    self.initializer)

                disc_vars.append(all_weights[name])
                #name = 'disc' + str(disc_i) + '_b' + str(layer_i)
                #all_weights[name] = ops.variable(name, 
                #        [cur_params['dim'][layer_i]], 
                #        tf.constant_initializer(0.0))
                #disc_vars.append(all_weights[name])

                name = 'disc_' + str(disc_i) + '_bn' + str(layer_i)
                batch_norms[name] = ops.batch_norm(cur_params['dim'][layer_i], name=name)

                prev_layer_dim = cur_params['dim'][layer_i]

        return all_weights, batch_norms, gen_vars, disc_vars


    def generator(self):
        prev_layer = self.z
        num_layers = len(self.generator_params['dim'])
        for layer_i in xrange(num_layers):
            conv = tf.nn.conv2d_transpose(prev_layer, self.all_weights['gen_w'+str(layer_i)], 
                output_shape=[self.batch_size, 
                      self.generator_params['shape'][layer_i], 
                      self.generator_params['shape'][layer_i], 
                      self.generator_params['dim'][layer_i]], 
                strides=[1, 2, 2, 1], padding='SAME')
            if layer_i+1==num_layers:
                lin = tf.nn.bias_add(conv, self.all_weights['gen_b'+str(layer_i)])
                nonlin = -0.1 + 1.2*tf.nn.sigmoid(lin)
            else:
                lin = conv
                linbn = self.batch_norms['gen_bn' + str(layer_i)](lin)
                nonlin = tf.nn.elu(linbn)
                prev_layer = nonlin
        return nonlin

    def discriminators_logits(self, input_batch):
        all_logits = []
        for disc_i in xrange(len(self.discriminators_params)):
            all_logits.append(self.discriminator_logits(disc_i, input_batch))
        all_logits = tf.pack(all_logits)
        logits = tf.reduce_sum(all_logits, reduction_indices=[0])
        return logits


    def discriminator_logits(self, disc_i, input_batch):
        cur_params = self.discriminators_params[disc_i]
        prev_layer = input_batch
        for layer_i in xrange(len(cur_params['dim'])):
            conv = tf.nn.conv2d(prev_layer, 
                self.all_weights['disc'+str(disc_i)+'_w'+str(layer_i)],
                strides=[1, 2, 2, 1], padding='VALID')
            #lin = tf.nn.bias_add(conv, self.all_weights['disc'+str(disc_i)+'_b'+str(layer_i)])
            lin = conv
            linbn = self.batch_norms['disc_' + str(disc_i) + '_bn' + str(layer_i)](lin)
            nonlin = tf.nn.elu(linbn)
            prev_layer = nonlin
        return tf.reshape(nonlin, [-1, 2])

    def get_losses(self):
        self.fake_images = self.generator()

        fake_disc_logits = self.discriminators_logits(self.fake_images)
        real_disc_logits = self.discriminators_logits(self.images)

        gen_loss = tf.nn.softmax_cross_entropy_with_logits(fake_disc_logits, self.true_labels)
        gen_loss = 2*tf.reduce_sum(gen_loss)

        real_disc_loss = tf.nn.softmax_cross_entropy_with_logits(
                real_disc_logits, self.true_labels)
        fake_disc_loss = tf.nn.softmax_cross_entropy_with_logits(
                fake_disc_logits, self.false_labels)

        disc_loss = tf.reduce_sum(real_disc_loss + fake_disc_loss)
        return gen_loss, disc_loss

    def train_steps(self, global_step):
        # Variables that affect learning rate.
        decay_steps = int(flags.NUM_STEPS_PER_DECAY)
        learning_rate_decay_factor = flags.LEARNING_RATE_DECAY_FACTOR
        moving_average_decay = flags.MOVING_AVERAGE_DECAY

        self.gen_loss, self.disc_loss = self.get_losses()

        # Compute gradients.
        with tf.control_dependencies([self.gen_loss, self.disc_loss]):
            gen_train_step_op = ops.train(
                    self.gen_loss, 
                    global_step, 
                    decay_steps, 
                    flags.GEN_INITIAL_LEARNING_RATE,
                    learning_rate_decay_factor, 
                    moving_average_decay, 
                    self.gen_vars, 
                    'gen')

            disc_train_step_op = ops.train(
                    self.disc_loss, 
                    global_step, 
                    decay_steps, 
                    flags.DISC_INITIAL_LEARNING_RATE,
                    learning_rate_decay_factor, 
                    moving_average_decay, 
                    self.disc_vars, 
                    'disc')

            return [gen_train_step_op, disc_train_step_op]

        

