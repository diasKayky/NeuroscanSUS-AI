# Arquivo: treinamento.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License

import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from modelo import NeuroscanSUS
from modelo_translearning import Neuroscan_TF

"""
Dados de Treinamento
"""

dados_treino = tf.data.Dataset.load("assets/imagens_treino")
dados_teste = tf.data.Dataset.load("assets/imagens_teste")
dados_val = tf.data.Dataset.load("assets/imagens_validacao")

"""
Modelo
"""

params = {"neuronios": 64, "entrada":(400, 400, 3), "lr": .0001,
          "kernel_size": (2, 2), "pool_size": (2, 2), "classes": 4}

modelo = NeuroscanSUS(params["neuronios"], params["entrada"],
                      params["kernel_size"], params["pool_size"], params["classes"])

modelo = modelo.constroi_modelo()

"""
Compilação
"""

loss = "sparse_categorical_crossentropy"
otimizador = tf.keras.optimizers.Adam(learning_rate=params["lr"])
epochs = 5

# Compila o modelo
modelo.compile(loss=loss, optimizer=otimizador, metrics=["accuracy"])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint('modelos/modelo7.h5', monitor='val_accuracy', save_best_only=True)

"""
Treinamento
"""
# Tempo inicial
inicial = datetime.now()

# Treinamento do modelo
history = modelo.fit(dados_treino, batch_size=32, epochs=epochs,
                     validation_data=dados_val, callbacks=[early_stopping, checkpoint])
# Tempo final
final = datetime.now()

tempo_total = final - inicial
print(f"O tempo que levou para treinar é de {tempo_total}")


