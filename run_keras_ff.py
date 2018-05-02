import numpy as np
import matplotlib.pyplot as plt
import os

from get_frey import *
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as tf
from keras import metrics
from keras.objectives import binary_crossentropy

# Configure matplotlib
#%matplotlib inline
plt.rcParams['figure.figsize'] = (13.5, 13.5) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

intermediate_dim = 256
img_rows = 28
img_cols = 20
n_pixels = img_rows * img_cols
latent_dim = 2
batch_size = 100
nb_epoch = 5
noise_std = .01

x = Input(shape=(n_pixels,)) #replace with n_pixels during frey_face
h = Dense(intermediate_dim, activation="relu")(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=noise_std)
    epsilon *= tf.exp(.5 * z_log_var)
    epsilon += z_mean
    return epsilon

def vae_objective(x, x_decoded):
    loss = binary_crossentropy(x, x_decoded)
    kl_regu = -.5 * tf.sum(1. + z_log_var - tf.square(
        z_mean) - tf.exp(z_log_var), axis=-1)
    return loss + kl_regu


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_h1 = Dense(intermediate_dim, activation="relu")
decoder_h2 = Dense(mnist_dim, activation="sigmoid")
z_decoded = decoder_h1(z)
x_decoded = decoder_h2(z_decoded)

# instantiate VAE model
vae = Model(input=x, output=x_decoded)
vae.compile(optimizer="adam", loss=vae_objective)
vae.summary()

weights_file = "ff_%d_latent.hdf5" % latent_dim
#if os.path.isfile(weights_file):
#	print "h"
#	vae.load_weights(weights_file)
#else:
from keras.callbacks import History
print "j"
hist_cb = History()
vae.fit(X_train, X_train, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size,
        callbacks=[hist_cb], validation_data=(X_val, X_val))
vae.save_weights(weights_file)

encoder = Model(input=x, output=z_mean)

decoder_input = Input(shape=(latent_dim,))
h_decoded = decoder_h1(decoder_input)
x_decoded = decoder_h2(h_decoded)
generator = Model(input=decoder_input, output=x_decoded)
    
# plot convergence curves to show off
plt.plot(hist_cb.history["loss"], label="training")
plt.plot(hist_cb.history["val_loss"], label="validation")
plt.grid("on")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="best")
print "here"
plt.show()
