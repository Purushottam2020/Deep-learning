# -*- coding: utf-8 -*-
"""Ex 3 Getting started with activation functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11W4j56dChFNnEuXI1oPDXan_XbOK9T1U
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import Callback

from keras.datasets import mnist

import random
SEED = random.randint(1,10000)

(X_train,y_train),(X_val, y_val) = mnist.load_data()

unique_labels = set(y_train)
plt.figure(figsize=(12,12))
i = 1
for label in unique_labels:
  image = X_train[y_train.tolist().index(label)]
  plt.subplot(10,10,i)
  plt.axis('off')
  plt.title("{0}: ({1})".format(label, y_train.tolist().count(label)))
  i += 1
  _ = plt.imshow(image, cmap = "gray")
plt.show()

X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255

y_train = to_categorical(y_train,10)
y_val = to_categorical(y_val,10)

X_train = np.reshape(X_train,(60000,784))
X_val = np.reshape(X_val, (10000,784))

model_sigmoid = Sequential()
model_sigmoid.add(Dense(700, input_dim=784,activation='sigmoid'))
model_sigmoid.add(Dense(700,activation='sigmoid'))
model_sigmoid.add(Dense(700,activation='sigmoid'))
model_sigmoid.add(Dense(700,activation='sigmoid'))
model_sigmoid.add(Dense(700,activation='sigmoid'))
model_sigmoid.add(Dense(350,activation='sigmoid'))
model_sigmoid.add(Dense(100,activation='sigmoid'))
model_sigmoid.add(Dense(10,activation='softmax'))

model_sigmoid.compile(loss='categorical_crossentropy',optimizer='sgd',metrics= ['accuracy'])

model_relu = Sequential()
model_relu.add(Dense(700, input_dim=784,activation='relu'))
model_relu.add(Dense(700,activation='relu'))
model_relu.add(Dense(700,activation='relu'))
model_relu.add(Dense(700,activation='relu'))
model_relu.add(Dense(700,activation='relu'))
model_relu.add(Dense(350,activation='relu'))
model_relu.add(Dense(100,activation='relu'))
model_relu.add(Dense(10,activation='softmax'))

model_relu.compile(loss='categorical_crossentropy',optimizer='sgd',metrics= ['accuracy'])

class history_loss(Callback):
  def on_train_begin(self,logs = {}):
    self.losses = []
  def on_batch_end(self, batch, logs = {}):
    batch_loss = logs.get('loss')
    self.losses.append(batch_loss)

n_epochs = 10
batch_size = 256
validation_split = 0.2
history_sigmoid = history_loss()
print("Name: G.Siva Ashok (9920004530)")
model_sigmoid.fit(X_train, y_train, epochs=n_epochs,batch_size=batch_size,callbacks=[history_sigmoid],validation_split=validation_split, verbose=2)

n_epochs = 10
batch_size = 256
validation_split = 0.2
history_relu = history_loss()
print("Name: G.Siva Ashok (9920004530)")
model_relu.fit(X_train, y_train, epochs=n_epochs,batch_size=batch_size,callbacks=[history_relu],validation_split=validation_split, verbose=2)

plt.plot(np.arange(len(history_sigmoid.losses)), history_sigmoid.losses, label = 'sigmoid')
plt.plot(np.arange(len(history_relu.losses)), history_relu.losses, label = 'relu')
plt.title("Name: G.Siva Ashok (9920004530)")
plt.suptitle('losses for sigmoid and relu model')
plt.xlabel("no of batches")
plt.ylabel("loss")
plt.legend(loc=1)
plt.show()

w_sigmoid =[]
w_relu = []
for i in range(len(model_sigmoid.layers)):
  w_sigmoid.append(max(model_sigmoid.layers[i].get_weights()[1]))
  w_relu.append(max(model_relu.layers[i].get_weights()[1]))

fig, ax = plt.subplots()
index = np.arange(len(model_sigmoid.layers))
bar_width = 0.35
plt.bar(index, w_sigmoid, bar_width, label = 'sigmoid', color = 'b', alpha = 0.4)
plt.bar(index+bar_width, w_relu, bar_width, label = 'relu', color = 'r', alpha = 0.4)
plt.title("Name: G.Siva Ashok (9920004530)")
plt.suptitle("Maximum weights across layers for sigmoid and relu activation functions")
plt.xlabel("Layer number")
plt.ylabel("maximum weight")
plt.legend(loc =0 )
plt.xticks(index+bar_width/2, np.arange(8))
plt.show()

