# Arquivo: treinamento.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License

import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from modelo import NeuroscanSUS
from preprocessamento import *

"""
Dados de Treinamento
"""

gerador = Gerador_Imagens("dados/Training", "dados/Testing")
dados_treino = gerador.imagens_treino()
dados_val = gerador.imagens_validacao()

x_treino, y_treino = [], []
for x, y in dados_treino:

    x_treino.append(x)
    y_treino.append(y)

x_treino = tf.concat(x_treino, axis=0)
y_treino = tf.concat(y_treino, axis=0)


x_val, y_val = [], []
for x, y in dados_val:

    x_val.append(x)
    y_val.append(y)

x_val = tf.concat(x_val, axis=0)
y_val = tf.concat(y_val, axis=0)

y_treino = tf.keras.utils.to_categorical(y_treino, 4)
y_val = tf.keras.utils.to_categorical(y_val, 4)
"""
Modelo
"""

params = {"neuronios": 64, "entrada":(400, 400, 3), "lr": .00001,
          "kernel_size": (2, 2), "pool_size": (2, 2), "classes": 4}

modelo = NeuroscanSUS(params["neuronios"], params["entrada"],
                      params["kernel_size"], params["pool_size"], params["classes"])

modelo = modelo.constroi_modelo()

"""
Compilação
"""

loss = "categorical_crossentropy"
otimizador = tf.keras.optimizers.Adam(learning_rate=params["lr"])
epochs = 15

# Compila o modelo
modelo.compile(loss=loss, optimizer=otimizador, metrics=["accuracy"])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=14)
checkpoint = tf.keras.callbacks.ModelCheckpoint('modelos/modelo3.h5', monitor='val_accuracy', save_best_only=True)


"""
Treinamento
"""
# Tempo inicial
inicial = datetime.now()

# Treinamento do modelo
history = modelo.fit(x_treino, y_treino, batch_size=32, epochs=epochs,
                     validation_data=[x_val, y_val], callbacks=[early_stopping, checkpoint])
# Tempo final
final = datetime.now()

tempo_total = final - inicial
print(f"O tempo que levou para treinar é de {tempo_total}")


