from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy.misc


import mnist as mnist_data
mnist = mnist_data.read_data_sets("/tmp/data/", one_hot=True)

from model import GAN

import flags

def save_images(images, target_dir=flags.train_dir, prefix=''):
    for i in xrange(images.shape[0]):
        name = prefix + str(i) + '.png'
        img = np.squeeze(images[i,:,:,:])
        scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(os.path.join(target_dir, name))

def train():

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        batch_size = 128

        gan = GAN(batch_size=batch_size)
        train_steps_op = gan.train_steps(global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()
        # Start running operations on the Graph.
        config = tf.ConfigProto(
            device_count = {'GPU': 0},
            log_device_placement=flags.log_device_placement
        )
        sess = tf.Session(config=config)

        summary_writer = tf.train.SummaryWriter(flags.train_dir, sess.graph)

        sess.run(tf.initialize_all_variables())

        num_steps = 5000 #int(mnist.train.num_examples/batch_size)

        for step in xrange(num_steps):
            start_time = time.time()

            train_images, _, _ = mnist.train.next_batch(batch_size)
            train_images = train_images.reshape([batch_size, 28, 28, 1])

            cur_feed_dict={gan.images: train_images}
            _ = sess.run(train_steps_op, feed_dict=cur_feed_dict)
            gen_loss_val, disc_loss_val = sess.run([gan.gen_loss, gan.disc_loss], feed_dict=cur_feed_dict)

            if step%2==1:
                fake_images = sess.run(gan.fake_images)
                save_images(fake_images, prefix='fake')

            if step%1==0 or (step + 1) == num_steps:
                format_str = ('%s: step %d of %d, gen_loss = %.5f, disc_loss = %.5f')
                print (format_str % (datetime.now(), step, num_steps-1, gen_loss_val, disc_loss_val))

            if step%10==0:
                summary_str = sess.run(summary_op, feed_dict=cur_feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step%50==0 or (step + 1)==num_steps:
                checkpoint_path = os.path.join(flags.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=(step))

def main(argv=None):  # pylint: disable=unused-argument
    if not os.path.exists(flags.train_dir):
        os.makedirs(flags.train_dir)
    
    train()

if __name__ == '__main__':
    tf.app.run()

