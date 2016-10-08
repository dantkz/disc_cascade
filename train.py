from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin

import mnist as mnist_data
mnist = mnist_data.read_data_sets("/tmp/data/", one_hot=True)

from model import GAN

def train():

  with tf.Graph().as_default():

    global_step = tf.Variable(0, trainable=False)

    batch_size = 128

    gan = GAN(batch_size = batch_size)

    train_step_op = gan.train_step(global_step)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
      with tf.device("/cpu:0"):

        summary_writer = tf.train.SummaryWriter('log/', sess.graph)

        sess.run(tf.initialize_all_variables())

        num_steps = int(mnist.train.num_examples/batch_size)

        for epoch in xrange(1):
          for step in xrange(num_steps):
            start_time = time.time()

            train_images, _, _ = mnist.train.next_batch(batch_size)
            train_images = train_images.reshape([batch_size, 28, 28, 1])
            print(train_images.shape)

            #_ = sess.run(train_step_op, feed_dict={gan.images: train_images})
            _ = sess.run(train_step_op, feed_dict={})


            if step%10==0 or (step + 1) == num_step:
              format_str = ('%s: epoch:%d/%d, step %d of %d, loss = %.5f')
              print (format_str % (datetime.now(), epoch, flags.max_epochs, step, num_step-1, loss_value))

            if (step + 1) == num_step:
              summary_str = sess.run(summary_op, feed_dict=cur_feed_dict)
              summary_writer.add_summary(summary_str, epoch)

            # Save the model checkpoint periodically.
            if (step + 1) == num_step:
              checkpoint_path = os.path.join(flags.train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=(epoch))

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()

