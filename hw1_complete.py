#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

def build_model1():
  model = Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(128),
    layers.LeakyReLU(),
    layers.Dense(128),
    layers.LeakyReLU(),
    layers.Dense(128),
    layers.LeakyReLU(),
    layers.Dense(10)  # logits
  ])

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model

def build_model2():
  model = Sequential([
    layers.Conv2D(32, 3, strides=2, padding='same', activation='relu',
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),

    layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
  ])

  for _ in range(4):
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
  model.add(layers.BatchNormalization())

  model.add(layers.Flatten())
  model.add(layers.Dense(10))

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )


  return model

def build_model3():
  inputs = Input(shape=(32, 32, 3))

  x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
  x = layers.BatchNormalization()(x)

  x = layers.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  for _ in range(4):
    x = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

  x = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.Flatten()(x)
  outputs = layers.Dense(10)(x)

  model = tf.keras.Model(inputs, outputs)

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  from keras.datasets import cifar10
  from sklearn.model_selection import train_test_split

  # Load CIFAR-10
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

  # Normalize images to [0,1]
  train_images = train_images.astype("float32") / 255.0
  test_images = test_images.astype("float32") / 255.0

  # Split training into train/validation
  train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
  )
  ########################################
  ## Build and train model 1
  model1 = build_model1()

  history1 = model1.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30,
    batch_size=64
  )

  test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
  # compile and train model 1.

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  history2 = model2.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=30,
    batch_size=64
  )

  test_loss2, test_acc2 = model2.evaluate(test_images, test_labels)
  
  ### Repeat for model 3 and your best sub-50k params model
  
  
