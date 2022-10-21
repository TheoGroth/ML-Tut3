#!/usr/bin/env python3

import sys
import nn
import pickle


save_path = sys.argv[1]

param_w = nn.init_mat(28 * 28, 10)
param_b = nn.init_vec(10)

f = open(save_path, 'wb')
pickle.dump((param_w, param_b), f)
f.close()
