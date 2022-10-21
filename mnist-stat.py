#!/usr/bin/env python3

import mnist
import numpy
import pickle

images = mnist.load_images('mnist/train-images-idx3-ubyte')
labels = mnist.load_labels('mnist/train-labels-idx1-ubyte')

ex = numpy.zeros(28 * 28)
ex2 = numpy.zeros(28 * 28)
samples = 0

for img, label in zip(images, labels):
    img = img.reshape(28 * 28)
    ex += img
    ex2 += img * img + 1e-8
    samples += 1

ex = ex / samples
ex2 = ex2 / samples

f = open('stat.npy', 'wb')
pickle.dump((ex, ex2), f)
f.close()
