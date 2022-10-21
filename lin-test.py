#!/usr/bin/env python3

import sys
import mnist
import nn
import numpy
import pickle

images = mnist.load_images('mnist/t10k-images-idx3-ubyte')
labels = mnist.load_labels('mnist/t10k-labels-idx1-ubyte')

f = open(sys.argv[1], 'rb')
param_w, param_b = pickle.load(f)
f.close()

zero_one_loss = 0
samples = 0

for img, label in zip(images, labels):
    img = mnist.normalize(img.reshape(28 * 28))

    g = nn.Graph()
    w = g.var(param_w)
    b = g.var(param_b)
    x = g.var(img)
    score = g.add(g.mul(x, w), b)

    nn.forward(g)

    pred, _ = max(enumerate(score.value), key=lambda t: t[1])

    zero_one_loss += (1 if pred != int(label) else 0)
    samples += 1

print('error count: {}, samples: {}, error rate: {:.6}'.format(zero_one_loss, samples, zero_one_loss / samples))
