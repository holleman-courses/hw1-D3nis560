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
    layers.Flatten(name="flatten", input_shape=(32, 32, 3)),
    layers.Dense(128, name="dense1"),
    layers.LeakyReLU(name="leaky_relu1"),
    layers.Dense(128, name="dense2"),
    layers.LeakyReLU(name="leaky_relu2"),
    layers.Dense(128, name="dense3"),
    layers.LeakyReLU(name="leaky_relu3"),
    layers.Dense(10, name="dense_out")  # logits
  ])

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )

  return model

def build_model2():
  model = Sequential([
    layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', name="conv2d_1", input_shape=(32, 32, 3)),
    layers.BatchNormalization(name="bn_1"),

    layers.Conv2D(64, 3, strides=2, padding='same', activation='relu', name="conv2d_2"),
    layers.BatchNormalization(name="bn_2"),
  ])

  for i in range(3, 7):
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu', name=f"conv2d_{i}"))
    model.add(layers.BatchNormalization(name=f"bn_{i}"))

  model.add(layers.Conv2D(128, 3, padding='same', activation='relu', name="conv2d_7"))
  model.add(layers.BatchNormalization(name="bn_7"))

  model.add(layers.Flatten(name="flatten"))
  model.add(layers.Dense(10, name="dense_out"))

  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  return model

def build_model3():
  inputs = Input(shape=(32, 32, 3), name="input")

  x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', name="conv2d_1")(inputs)
  x = layers.BatchNormalization(name="bn_1")(x)

  x = layers.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu', name="sep_conv2d_2")(x)
  x = layers.BatchNormalization(name="bn_2")(x)

  for i in range(3, 7):
    x = layers.SeparableConv2D(64, 3, padding='same', activation='relu', name=f"sep_conv2d_{i}")(x)
    x = layers.BatchNormalization(name=f"bn_{i}")(x)

  x = layers.SeparableConv2D(128, 3, padding='same', activation='relu', name="sep_conv2d_7")(x)
  x = layers.BatchNormalization(name="bn_7")(x)

  x = layers.Flatten(name="flatten")(x)
  outputs = layers.Dense(10, name="dense_out")(x)

  model = tf.keras.Model(inputs, outputs)
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
  return model

def build_model50k():
  inputs = Input(shape=(32, 32, 3), name="input")

  x = layers.SeparableConv2D(32, 3, strides=2, padding='same', activation='relu', name="sep_conv2d_1")(inputs)
  x = layers.BatchNormalization(name="bn_1")(x)

  x = layers.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu', name="sep_conv2d_2")(x)
  x = layers.BatchNormalization(name="bn_2")(x)

  x = layers.SeparableConv2D(64, 3, padding='same', activation='relu', name="sep_conv2d_3")(x)
  x = layers.BatchNormalization(name="bn_3")(x)

  x = layers.GlobalAveragePooling2D(name="gap")(x)
  outputs = layers.Dense(10, name="dense_out")(x)

  model = tf.keras.Model(inputs, outputs)
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
  )
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
  
  
