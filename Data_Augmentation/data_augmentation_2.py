#CIFAR 10 veri kümesi ile veri arttırma
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras import backend as k
import numpy as np

#Verinin eğitim ve test kümelerini rastgele oluşturalım
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#görüntüler veri setinde 255,255
x_train /= 255
x_test /= 255

#orijinal görüntüler
datagen = ImageDataGenerator()
datagen.fit(x_train)

#batch_size aynı anda kaç görüntü alacağını belirlemek için
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 + 1 + i )
        plt.imshow(x_batch[i])
    plt.show()
    break

#Döndürme Yöntemi
datagen = ImageDataGenerator(rotation_range=359)
datagen.fit(x_train)
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 + 1 + i )
        plt.imshow(x_batch[i])
    plt.show()
    break

#Dikey Eksende Kaydırma Yöntemi
datagen = ImageDataGenerator(height_shift_range=0.6)
datagen.fit(x_train)
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 + 1 + i )
        plt.imshow(x_batch[i])
    plt.show()
    break

#Yatayda Simetriğini Alma
datagen = ImageDataGenerator(horizontal_flip=True)
datagen.fit(x_train)
for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=4, seed=499):
    for i in range(0,4):
        plt.subplot(220 + 1 + i )
        plt.imshow(x_batch[i])
    plt.show()
    break