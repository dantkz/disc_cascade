import numpy as np
import tensorflow as tf

import mnist as mnist_data
mnist = mnist_data.read_data_sets("/tmp/data/", one_hot=True)

from model import GAN

gan = GAN()

gan.train()
