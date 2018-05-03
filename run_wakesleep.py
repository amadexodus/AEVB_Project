import tensorflow as tf
import numpy as np
from get_mnist import *
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as tf
from keras import metrics

mnist_dim = 784
n_epochs = 5

x = tf.placeholder(np.float32, shape=(None, mnist_dim))
x = Input(shape=(mnist_dim,))