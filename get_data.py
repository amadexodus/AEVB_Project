import scipy.io
mnist_data = scipy.io.loadmat('mnist_all.mat')

#print mat.keys()

mnist_keys = ['test1', 'test0', 'test3', 'test2', 'test5', 'test4', 'test7', 'test6', 'test9', 'test8', 'train4', 'train5', 'train6', 'train7', 'train0', 'train1', 'train2', 'train3', '__version__', 'train8', 'train9', '__header__', '__globals__']

# type(mnist['test1']) = 'numpy.ndarray'
# type(mnist['test1'][0]) = 'numpy.ndarray'
# type(mnist['test1'][0][0]) = 'numpy.uint8'
#print type(mnist_data['test1'][0][0])

ff_data = scipy.io.loadmat('frey_rawface.mat')
ff = ff_data['ff']

#print ff_data.keys()
print type(ff)
print max(ff[0])