# Arquivo: preprocessamento.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)  
# 2023 - MIT License

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


class Gerador_Imagens:
    """
    Classe que constrói o gerador de imagens de treino e teste
    """
    def __init__(self, diretorio_treino, diretorio_teste, tamanho=(400, 400), batch_size=32):

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
            validation_split=.2,
            subset="training",
            seed=45,
            image_size=self.tamanho,
            batch_size=self.batch_size
        )

        return imagens_teste

    def imagens_validacao(self):

        # Imagens de Validação
        imagens_validacao = image_dataset_from_directory(
            self.diretorio_teste,
            validation_split=.2,
            seed=45,
            subset="validation",
            image_size=self.tamanho,
            batch_size=self.batch_size
        )

        return imagens_validacao



def main():

    # Gerador
    gerador = Gerador_Imagens("dados/Training", "dados/Testing")

    # Imagens de Treino
    imagens_treino = gerador.imagens_treino()
    imagens_treino.save("assets/imagens_treino")

    # Imagens de Teste
    imagens_teste = gerador.imagens_teste()
    imagens_teste.save("assets/imagens_teste")

    # Imagens de Validação
    imagens_validacao = gerador.imagens_validacao()
    imagens_validacao.save("assets/imagens_validacao")

if __name__ == "__main__":
    main()