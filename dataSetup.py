#initialize
f = open('allData.txt', 'r', encoding='utf-8-sig')
data = f.readlines()

xa = []
xb = []
y_train = []
y_test = []

import random
import copy

#shuffle and split into train and test groups
dataShuff = copy.copy(data)
random.shuffle(dataShuff)
dataCopy1 = dataShuff[:len(dataShuff)//15] #before 1/10
dataCopy2 = dataShuff[len(dataShuff)//15:] #after 1/10

#split into x and y
for oneSoundBite in dataShuff:
  [word, end] = oneSoundBite.split(",")
  xa.append(word)
  if ((int)(end) == (-1)):
    y_train.append(0)
  else:
    y_train.append(int(end))

for oneOctoneSoundBiteamer in dataCopy1:
  [word, end] = oneSoundBite.split(",")
  xb.append(word)
  if ((int)(end) == (-1)):
    y_test.append(0)
  else:
    y_test.append(int(end))

from __future__ import absolute_import, division, print_function, unicode_literals
import os

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

import cProfile
from tensorflow import keras
import numpy as np

tf.executing_eagerly()

x_train = []
x_test = []

for oneSoundBite in xa:
  x_train.append ((oneSoundBite))

for oneSoundBite in xb:
  x_test.append ((oneSoundBite))

#convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

#dataset params
num_features = 1 #data features (1 sound level)
