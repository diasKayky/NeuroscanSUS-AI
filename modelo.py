# Arquivo: modelo.py - Projeto Neuroscan.AI
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License

import pandas as pd
import numpy as np
import tensorflow as tf

class Neuroscan_AI:

    """
    Classe que armazena o modelo Neuroscan.AI
    """

    def __init__(self, neuronios, entrada, kernel_size, ativacao="relu", pool_size):

        self.neuronios = neuronios
        self.entrada = entrada
        self.kernel_size = kernel_size
        self.ativacao = ativacao
        self.pool_size = pool_size

    def constroi_modelo(self):

        return None


