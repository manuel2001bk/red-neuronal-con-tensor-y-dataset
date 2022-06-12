from ast import Try
from typing_extensions import Self
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from ventana_ui import *


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.entrenar.clicked.connect(self.entrenar_tensor)
        self.errores.clicked.connect(self.tabla_errores)
        self.predicciones.clicked.connect(self.tabla_predicciones)
        self.entrenamiento = 0

        self.categorias = []
        self.labels = []
        self.imagenes = []
        self.model = []
        self.historial = []
        self.predic = []

    def entrenar_tensor(self):
        self.limpieza()
        self.get_categorias()
        self.get_dataset()
        self.convertir_categorias()
        self.convertir_dataset()
        # print(self.imagenes.shape)
        self.get_modelo_red()
        self.compilar_modelo_red()
        self.entrenamiento = int(self.cant_entrenamiento.text())
        self.historial = self.model.fit(
            self.imagenes, self.labels, epochs=self.entrenamiento)

    def limpieza(self):
        self.categorias = []
        self.labels = []
        self.imagenes = []
        self.model = []
        self.historial = []
        self.predic = []

    def get_modelo_red(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    def compilar_modelo_red(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def get_imagen_ejemplo(self):
        plt.figure()
        plt.imshow(self.imagenes[15])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def convertir_categorias(self):
        print(self.labels)
        self.labels = np.asarray(self.labels)

    def convertir_dataset(self):
        print(self.labels)
        self.imagenes = np.asanyarray(self.imagenes)

    def get_dataset(self):
        x = 0
        for directorio in self.categorias:
            for imagen in os.listdir(f'Dataset/{directorio}'):
                img = Image.open(
                    f'Dataset/{directorio}/{imagen}').resize((28, 28))
                img = np.asarray(img)
                self.imagenes.append(img)
                self.labels.append(x)
            x += 1

    def get_categorias(self):
        self.categorias = os.listdir(f'Dataset')
        print(self.categorias)

    def tabla_errores(self):
        plt.plot(self.historial.history['loss'])
        plt.show()

    def tabla_predicciones(self):
        imge = Image.open(f'test/0.png')
        imge = np.asarray(imge)
        imge = np.array([imge])
        print(imge.shape)
        self.predic = self.model.predict(imge)

        print(self.categorias[np.argmax(self.predic[0])])


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
