# Arquivo: modelo.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License

import pandas as pd
import numpy as np
import tensorflow as tf

class NeuroscanSUS:

    """
    Classe que armazena o modelo NeuroscanSUS
    """

    def __init__(self, neuronios, entrada, kernel_size, pool_size, classes, ativacao="relu"):

        self.neuronios = neuronios
        self.entrada = entrada
        self.kernel_size = kernel_size
        self.classes = classes
        self.ativacao = ativacao
        self.pool_size = pool_size

    def constroi_modelo(self):

        modelo = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(self.neuronios, input_shape=self.entrada,
                                   kernel_size=self.kernel_size, activation=self.ativacao),
            tf.keras.layers.MaxPool2D(pool_size=self.pool_size),
            tf.keras.layers.Conv2D(self.neuronios, input_shape=self.entrada,
                                   kernel_size=self.kernel_size, activation=self.ativacao),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.classes, activation="softmax")
        ])

        return modelo




