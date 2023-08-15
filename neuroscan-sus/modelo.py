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

        input_1 = tf.keras.Input(shape=((400, 400, 3)))
        #input_2 = tf.keras.layers.experimental.preprocessing.Resizing(120, 120)(input_1)
        input_3 = tf.keras.layers.Normalization()(input_1)

        conv1 = tf.keras.layers.Conv2D(self.neuronios, input_shape=self.entrada,
                               kernel_size=self.kernel_size, activation=self.ativacao)(input_3)

        pool1 = tf.keras.layers.MaxPool2D(pool_size=self.pool_size)(conv1)
        norm1 = tf.keras.layers.BatchNormalization()(pool1)

        attn_1 = tf.keras.layers.GlobalAveragePooling2D()(norm1)
        attn_2 = tf.keras.layers.Dense(self.neuronios, activation='softmax')(attn_1)
        attn_3 = tf.keras.layers.Reshape((1, 1, self.neuronios))(attn_2)
        attention = tf.keras.layers.Multiply()([norm1, attn_3])

        conv2 = tf.keras.layers.Conv2D(self.neuronios, input_shape=self.entrada,
                               kernel_size=self.kernel_size, activation=self.ativacao)(attention)
        norm2 = tf.keras.layers.BatchNormalization()(conv2)
        outputs = tf.keras.layers.Activation(tf.keras.activations.relu)(norm2)


        outputs = tf.keras.layers.Flatten()(outputs)
        outputs = tf.keras.layers.Dense(self.classes, activation="softmax")(outputs)

        modelo = tf.keras.models.Model(inputs=input_1, outputs=outputs)

        return modelo




