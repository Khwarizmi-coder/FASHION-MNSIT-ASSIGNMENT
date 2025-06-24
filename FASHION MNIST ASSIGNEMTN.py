import tensorflow as tf
import keras
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Load Fashion MNIST
fashion_mnist=tf.keras.datasets.fashion_mnist
keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Normalize
x_train=x_train/255.0
x_test=x_test/255.0
#Reshape to add chanel
x_train=x_train[...,np.newaxis]
x_test=x_test[...,np.newaxis]
#Build CNN model with 6 layers
model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Training of the model
model.fit(x_train,y_train,epochs=5,validation_split=0.1)
#predict on two test images
predictions=model.predict(x_test[:2])
predicted_classes=np.argmax(predictions,axis=1)
#plot images and predictions
for i in range(2):
    plt.imshow(x_test[i].reshape(28,28),cmap='gray')
    plt.title(f"Predicted:{predicted_classes[i]}")
    plt.show()