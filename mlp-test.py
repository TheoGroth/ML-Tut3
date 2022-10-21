#!/usr/bin/env python3

import sys
import mnist
import nn
import numpy
import pickle

images = mnist.load_images('mnist/t10k-images-idx3-ubyte')
labels = mnist.load_labels('mnist/t10k-labels-idx1-ubyte')

f = open(sys.argv[1], 'rb')
param_w1, param_b1, param_w2, param_b2 = pickle.load(f)
f.close()

zero_one_loss = 0
samples = 0

for img, label in zip(images, labels):
    img = mnist.normalize(img.reshape(28 * 28))

    g = nn.Graph()
    w1 = g.var(param_w1)
    b1 = g.var(param_b1)
    w2 = g.var(param_w2)
    b2 = g.var(param_b2)
    x = g.var(img.reshape(28 * 28))
    score = g.add(g.mul(g.logistic(g.add(g.mul(x, w1), b1)), w2), b2)

    nn.forward(g)

    pred, _ = max(enumerate(score.value), key=lambda t: t[1])

    zero_one_loss += (1 if pred != int(label) else 0)
    samples += 1

print('error count: {}, samples: {}, error rate: {:.6}'.format(zero_one_loss, samples, zero_one_loss / samples))
