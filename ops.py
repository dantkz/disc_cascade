import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as tftensor

class batch_norm(object):
    def __init__(self, xdim, epsilon=1e-5, momentum = 0.9, name=''):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.pop_mean = variable('pop_mean', [xdim], tf.constant_initializer(0.0), trainable=False)
            self.pop_var = variable('pop_var', [xdim], tf.constant_initializer(1.0), trainable=False)
            self.beta = variable('beta', [xdim], tf.constant_initializer(0.0))
            self.gamma = variable('gamma', [xdim], tf.constant_initializer(1.0))


    def __call__(self, x, istrain=True):
        if istrain:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema_apply_op = self.ema.apply([batch_mean, batch_var])
            pop_mean_op = tf.assign(self.pop_mean, self.ema.average(batch_mean))
            pop_var_op = tf.assign(self.pop_var, self.ema.average(batch_var))

            with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
                mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.pop_mean, self.pop_var

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
       
        return normed

 
def downconv_layer(inp, ksize, inpsize, outsize, stride=2, bias_init = 0.0, add_bias=True, conv_std=1e-4):
    kernel = variable_with_weight_decay('weights', shape=[ksize, ksize, inpsize, outsize],
                                         stddev=conv_std, wd=0.0)
    conv = tf.nn.conv2d(inp, kernel, [1, stride, stride, 1], padding='SAME')
    if add_bias:
        biases = variable('biases', [outsize], tf.constant_initializer(bias_init))
        out = tf.nn.bias_add(conv, biases)
    else:
        out = conv
    return out

def upconv_layer(inp, kernel, inpsize, outsize, outshape, stride=2, add_bias=True):
    kernel = variable_with_weight_decay('weights', shape=[ksize, ksize, outsize, inpsize],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d_transpose(inp, kernel, output_shape=[inp.get_shape().as_list()[0], outshape, outshape, outsize], strides=[1, stride, stride, 1], padding='SAME')
    if add_bias:
        biases = variable('biases', [outsize], tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases)
    else:
        out = conv
    return out


def variable(name, shape=None, initializer=tf.constant_initializer(0.0), trainable=True):
    """Helper to create a Variable stored on GPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        if shape is None:
            var = tf.get_variable(name, initializer=initializer, trainable=trainable)
        else:
            var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    return var



def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    var = variable(name, shape,
                           tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
                           #tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None and wd>0.0:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def train(total_loss, global_step, decay_steps, initial_learning_rate, learning_rate_decay_factor, moving_average_decay, beta1=0.5, target_vars=tf.trainable_variables(), name='all'):
    """Train the model.
  
    Create an optimizer and apply to target trainable variables. Add moving
    average for target trainable variables.
  
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.scalar_summary('learning_rate_'+name, lr)

    opt = tf.train.AdamOptimizer(lr, beta1=beta1)
    grads = opt.compute_gradients(total_loss, var_list=target_vars)
  
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
    ## Add histograms for trainable variables.
    #for var in target_vars:
    #    tf.histogram_summary(var.op.name, var)
  
    ## Add histograms for gradients.
    #for grad, var in grads:
    #    if grad is not None:
    #        tf.histogram_summary(var.op.name + '/gradients', grad)
  
    # Track the moving averages of target trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(target_vars)
  
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train_' + name)
  
    return train_op

