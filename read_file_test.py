# -*- coding: utf-8 -*-

'''
  Copyright(c) 2018, LiuYang
  All rights reserved.
  2018/02/23
'''

import csv
import numpy as np
import pickle
path = r'test_data\test_batch'
fo = open(path, 'rb')
dicts = pickle.load(fo, encoding='bytes')
fo.close()
data = np.array(dicts[b'data'])
label = np.array(dicts[b'labels'])
''''
filepath = r'D:\program\liuyang\liuyang\interpreting_data\cifar10\data\csv_file\validate_5000.csv'
features = []
labels = []
with open(filepath, 'r') as f:
    for line in f.readlines():
        temp = [ele for ele in line.strip().split(',')]
        features.append(temp[0])
        labels.append(int(temp[1]))

features = np.array(features)
labels = np.array(labels)
order = np.random.permutation(len(labels))
features = features[order]
labels = labels[order]
batch_feature = []
batch_label = []
print(features[0])
print(labels[0])
print(order)
'''