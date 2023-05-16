# Arquivo: avaliacao.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License

import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from preprocessamento import *

"""
Assets
"""
# Modelo treinado 1
modelo = tf.keras.models.load_model("modelos/modelo2.h5")

# Dados de Teste
gerador = Gerador_Imagens("dados/Training", "dados/Testing")
dados_teste = gerador.imagens_teste()
x, y = next(iter(dados_teste))

# Mapa de Classes
class_map = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}


"""
Predição
"""
pred = np.argmax(modelo.predict(x), axis=-1)


"""
Avaliação
"""
clf_report = classification_report(y, pred, target_names=["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"])
cf_matrix = confusion_matrix(y, pred, labels=[0, 1, 2, 3])

print(clf_report)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
plt.show()