import scipy.io
import numpy as np

path = "data/mnist/mnist_all.mat"

mnist = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

print(mnist.keys())

X_train = np.concatenate((mnist['train0'], mnist['train1'], mnist['train2'], mnist['train3'],
	mnist['train4'], mnist['train5'], mnist['train6'], mnist['train7'], mnist['train8'],
	mnist['train9']), axis=0)
X_test = np.concatenate((mnist['test0'], mnist['test1'], mnist['test2'], mnist['test3'],
	mnist['test4'], mnist['test5'], mnist['test6'], mnist['test7'], mnist['test8'],
	mnist['test9']), axis=0)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

print X_test.shape[0]
print X_train.shape[0]

