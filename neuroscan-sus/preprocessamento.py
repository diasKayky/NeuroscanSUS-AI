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

class Data_Augmentation:
    """
    Classe que faz aumento nos dados
    """
    def flip_image(self, image, label):

        flipped_image = tf.image.flip_left_right(image)
        return flipped_image, label

    def adjust_brightness(self, image, label):

        bright_image = tf.image.adjust_brightness(image, delta=0.2)
        return bright_image, label

    def rotate_image(self, image, label):

        rotated_image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return rotated_image, label

    def apply_zoom(self, image, label):

        zoomed_image = tf.image.central_crop(image, central_fraction=0.8)
        image_shape = tf.shape(image)
        zoomed_image = tf.image.resize(zoomed_image, size=[image_shape[0], image_shape[1]])
        return zoomed_image, label

def main():

    # Gerador e Data Aug
    gerador = Gerador_Imagens("dados/Training", "dados/Testing")
    data_aug = Data_Augmentation()

    # Imagens de Treino + Data Augmentation
    imagens_treino = gerador.imagens_treino()
    augmented_dataset = imagens_treino.map(data_aug.flip_image)
    augmented_dataset = augmented_dataset.concatenate(imagens_treino.map(data_aug.adjust_brightness))
    augmented_dataset = augmented_dataset.concatenate(imagens_treino.map(data_aug.rotate_image))
    augmented_dataset = augmented_dataset.concatenate(imagens_treino.map(data_aug.apply_zoom))

    augmented_dataset.save("assets/imagens_treino")

    # Imagens de Teste
    imagens_teste = gerador.imagens_teste()
    imagens_teste.save("assets/imagens_teste")

    # Imagens de Validação
    imagens_validacao = gerador.imagens_validacao()
    imagens_validacao.save("assets/imagens_validacao")

if __name__ == "__main__":
    main()