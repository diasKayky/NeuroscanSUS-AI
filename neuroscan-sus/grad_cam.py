# Arquivo: grad_cam.py - Projeto NeuroscanSUS
# Autor: Kayky Santos (https://github.com/diasKayky)
# 2023 - MIT License

# Importações de libraries importantes
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """
    Classe que armazena o GradCAM: Mapeamento de Ativação por Pesos do Gradiente
    """

    def __init__(self, modelo, ultima_conv):
        # Instâncias da classe
        self.modelo = modelo
        self.ultima_conv = ultima_conv
        self.grad_modelo = tf.keras.models.Model(
            [self.modelo.inputs], [self.modelo.get_layer(self.ultima_conv).output, self.modelo.output]
        )

    def preprocessa_input(self, x):
        # Preprocessa a entrada

        x = tf.keras.applications.resnet50.preprocess_input(x)

        return x

    def calcula_gradcam(self, imagem):

        imagem = tf.expand_dims(imagem, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predicoes = self.grad_modelo(imagem)
            top_predicao = tf.argmax(predicoes[0])
            top_class = predicoes[:, top_predicao]

        grads = tape.gradient(top_class, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap
        max_value = np.max(heatmap)
        if max_value != 0:
            heatmap /= max_value


        return heatmap

    def faz_gradcam(self, path_da_imagem=""):

        imagem = cv2.imread(path_da_imagem)

        if imagem is None:
            print(f"Falha ao carregar a imagem: {path_da_imagem}")
            return 0

        # Preprocess the image
        imagem = cv2.resize(imagem, (400, 400))

        # Calculate GradCAM
        heatmap = self.calcula_gradcam(imagem)

        # Normalize the heatmap
        heatmap = cv2.resize(heatmap, (imagem.shape[1], imagem.shape[0]))
        heatmap = (heatmap * 255).astype(np.uint8)

        print(imagem.shape)
        print(imagem.dtype)

        # Aplica o heatmap na imagem original
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HSV)
        imagem_com_gradcam = cv2.addWeighted(imagem, 0.65, heatmap, 0.35, 0)

        # Mostra imagem
        plt.imshow(imagem_com_gradcam)
        plt.axis("off")
        plt.show()
