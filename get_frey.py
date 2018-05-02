import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

img_rows = 28
img_cols = 20

ff = scipy.io.loadmat("data/freyface/frey_rawface.mat", squeeze_me=True, struct_as_record=False)
ff = ff["ff"].T.reshape((-1, img_rows, img_cols))

# split datainto train/validation folds
np.random.seed(42)
n_pixels = img_rows * img_cols
X_train = ff[:1800]
X_val = ff[1800:1900]
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_train = X_train.reshape((len(X_train), n_pixels))
X_val = X_val.reshape((len(X_val), n_pixels))


'''
ff = scipy.io.loadmat('frey_rawface.mat')['ff']

tensor_ff = tf.convert_to_tensor(ff)

print len(ff[559])

print tensor_ff
'''
