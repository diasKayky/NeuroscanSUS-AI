# Arquivo: modelo_translearning.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50


class Neuroscan_TF:

    def __init__(self, neuronios, input_shape, classes):

        self.neuronios = neuronios
        self.input_shape = input_shape
        self.classes = classes

    def constroi_modelo(self):

        entradas = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Rescaling(1.0 / 255)(entradas)

        modelo_base = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape)
        modelo_base.trainable = False

        x = modelo_base(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.neuronios, activation="relu")(x)
        predicao = tf.keras.layers.Dense(self.classes, activation="softmax")(x)
        modelo = tf.keras.models.Model(inputs=entradas, outputs=predicao)

        return modelo
