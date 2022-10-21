#!/usr/bin/env python3

import sys
import mnist
import nn
import numpy
import pickle


def train_epoch(param_w1, param_b1, param_w2, param_b2):
    images = mnist.load_images('mnist/train-images-idx3-ubyte')
    labels = mnist.load_labels('mnist/train-labels-idx1-ubyte')

    step_size = 0.01

    total_loss = 0
    samples = 0

    for img, label in zip(images, labels):
        img = mnist.normalize(img.reshape(28 * 28))

        g = nn.Graph()
        w1 = g.var(param_w1)
        b1 = g.var(param_b1)
        w2 = g.var(param_w2)
        b2 = g.var(param_b2)
        x = g.var(img)
        y = g.var(nn.one_hot(label, 10))
        score = g.add(g.mul(g.logistic(g.add(g.mul(x, w1), b1)), w2), b2)
        loss = g.cross_entropy(score, y)
    
        nn.forward(g)
    
        print('sample: {}'.format(samples))
        print('loss: {:.6}'.format(loss.value))
    
        loss.grad = 1.0
        nn.backward(g)
    
        grad_norm = (numpy.linalg.norm(w1.grad) + numpy.linalg.norm(b1.grad)
            + numpy.linalg.norm(w2.grad) + numpy.linalg.norm(b2.grad))

        print('grad norm: {:.6}'.format(grad_norm))
    
        # gradient clipping
        if grad_norm > 5:
            param_w1 = param_w1 - step_size * w1.grad * 5 / grad_norm
            param_b1 = param_b1 - step_size * b1.grad * 5 / grad_norm
            param_w2 = param_w2 - step_size * w2.grad * 5 / grad_norm
            param_b2 = param_b2 - step_size * b2.grad * 5 / grad_norm
        else:
            param_w1 = param_w1 - step_size * w1.grad
            param_b1 = param_b1 - step_size * b1.grad
            param_w2 = param_w2 - step_size * w2.grad
            param_b2 = param_b2 - step_size * b2.grad

        total_loss += loss.value
        samples += 1
    
        print()

    return param_w1, param_b1, param_w2, param_b2, total_loss, samples


param_path = sys.argv[1]
save_path = sys.argv[2]

f = open(param_path, 'rb')
param_w1, param_b1, param_w2, param_b2 = pickle.load(f)
f.close()

param_w1, param_b1, param_w2, param_b2, total_loss, samples = train_epoch(param_w1, param_b1, param_w2, param_b2)

f = open(save_path, 'wb')
pickle.dump((param_w1, param_b1, param_w2, param_b2), f)
f.close()
