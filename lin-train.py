#!/usr/bin/env python3

import sys
import mnist
import nn
import numpy
import pickle


def train_epoch(param_w, param_b):
    images = mnist.load_images('mnist/train-images-idx3-ubyte')
    labels = mnist.load_labels('mnist/train-labels-idx1-ubyte')

    step_size = 0.01

    total_loss = 0
    samples = 0

    for img, label in zip(images, labels):
        img = mnist.normalize(img.reshape(28 * 28))

        g = nn.Graph()
        w = g.var(param_w)
        b = g.var(param_b)
        x = g.var(img)
        y = g.var(nn.one_hot(label, 10))
        score = g.add(g.mul(x, w), b)
        loss = g.cross_entropy(score, y)
    
        nn.forward(g)
    
        print('sample: {}'.format(samples))
        print('loss: {:.6}'.format(loss.value))
    
        loss.grad = 1.0
        nn.backward(g)
    
        grad_norm = numpy.linalg.norm(w.grad) + numpy.linalg.norm(b.grad)
        print('grad norm: {:.6}'.format(grad_norm))
    
        # gradient clipping
        if grad_norm > 5:
            param_w = param_w - step_size * w.grad * 5 / grad_norm
            param_b = param_b - step_size * b.grad * 5 / grad_norm
        else:
            param_w = param_w - step_size * w.grad
            param_b = param_b - step_size * b.grad

        total_loss += loss.value
        samples += 1
    
        print()

    return param_w, param_b, total_loss, samples


param_path = sys.argv[1]
save_path = sys.argv[2]

f = open(param_path, 'rb')
param_w, param_b = pickle.load(f)
f.close()

param_w, param_b, total_loss, samples = train_epoch(param_w, param_b)

f = open(save_path, 'wb')
pickle.dump((param_w, param_b), f)
f.close()

