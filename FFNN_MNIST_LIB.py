# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:52:53 2022

@author: craig
"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# =============================================================================
# (x_train, y_train), (x_val, y_val) = mnist.load_data()
# 
# x_train = x_train.astype('float32') / 255
# y_train = to_categorical(y_train)
# =============================================================================
train_dataset = pd.read_csv('mnist_train.csv')
test_dataset = pd.read_csv('mnist_test.csv')
X_train = train_dataset.drop(columns='label')
X_test = test_dataset.drop(columns='label')
Y_train = pd.get_dummies(train_dataset['label'])
y_test = test_dataset['label']
Y_test = pd.get_dummies(y_test)

model = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(10)
])

model.compile(optimizer='SGD',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10)

performance = model.evaluate(X_test, Y_test, verbose=1)
# print("Accuracy: ", performance_metric.append(performance[1] * 100))
# print("Mean: ",np.mean(performance_metric))
# print("Standard Deviation: ", np.std(performance_metric))
Result = model.predict(X_test)
# =============================================================================
# Confusion Matrix
# =============================================================================
Confusion = confusion_matrix(y_test,pd.Series(np.argmax(Result,axis = 1)))
plt.xlabel('Predicted class')
plt.ylabel('Occurence')
sns.heatmap(Confusion,annot = True,fmt = 'd')
plt.show()