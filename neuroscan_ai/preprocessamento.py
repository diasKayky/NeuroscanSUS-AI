# Arquivo: preprocessamento.py - Projeto Neuroscan.AI
# Autor: Kayky Santos (https://github.com/diasKayky)  
# 2023 - MIT License

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


class Gerador_Imagens:
    """
    Classe que constroí o gerador de imagens de treino e teste
    """
    def __init__(self, diretorio_treino, diretorio_teste, tamanho=(400, 400), batch_size=64):

        self.diretorio_treino = diretorio_treino
        self.diretorio_teste = diretorio_teste
        self.tamanho = tamanho
        self.batch_size = batch_size

    def imagens_treino(self):

        # Imagens de Treino
        imagens_treino = image_dataset_from_directory(
            self.diretorio_treino,
            image_size=self.tamanho,
            batch_size=self.batch_size
        )

        return imagens_treino

    def imagens_teste(self):

        # Imagens de Teste
        imagens_teste = image_dataset_from_directory(
            self.diretorio_teste,
            image_size=self.tamanho,
            batch_size=self.batch_size
        )

        return imagens_teste



def main():

    # Gerador
    gerador = Gerador_Imagens("dados/Training", "dados/Testing")

    # Imagens de Treino
    imagens_treino = gerador.imagens_treino()
    imagens_treino.save("assets/imagens_treino")

    # Imagens de Teste
    imagens_teste = gerador.imagens_teste()
    imagens_teste.save("assets/imagens_teste")

if __name__ == "__main__":
    main()