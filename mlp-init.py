#!/usr/bin/env python3

import sys
import nn
import pickle


save_path = sys.argv[1]

param_w1 = nn.init_mat(28 * 28, 128)
param_b1 = nn.init_vec(128)
param_w2 = nn.init_mat(128, 10)
param_b2 = nn.init_vec(10)

f = open(save_path, 'wb')
pickle.dump((param_w1, param_b1, param_w2, param_b2), f)
f.close()
