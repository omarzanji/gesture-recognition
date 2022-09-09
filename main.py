from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mediapipe as mp

from hand_tracking import HandTracking

class GestureNet:

    def __init__(self, train=False):
        self.labels = ['like', 'dislike', 'palm']
        if train:
            try:
                self.x = np.load('data/x_data.npy')
                self.y = np.load('data/y_data.npy')
            except:
                print('cached x and y arrays not found...')
        else:
            self.model = keras.models.load_model('models/GestureNet')
    
    def load_data(self):
        self.x_train = self.x
        self.y_train = []
        for label in self.y:
            onehot = [0, 0, 0]
            onehot[self.labels.index(label)] = 1
            self.y_train.append(onehot)
        self.y_train = np.array(self.y_train)

    def create_model(self):

        # AlexNet 
        model = Sequential([
            # layers.Normalization(),
            layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax'),
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
        return model

    def train_model(self, model):
        name = 'GestureNet'
        xtrain, xtest, ytrain, ytest = train_test_split(self.x_train, self.y_train, train_size=0.9)
        self.hist = model.fit(xtrain, ytrain, epochs=90, verbose=1)
        self.plot_history(self.hist)
        model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = ypreds
        accuracy = accuracy_score(ytest, np.round(self.ypreds).astype(int))
        print(f'\n\n[Accuracy: {accuracy}]\n\n')
        self.ytest = ytest
        model.save(f'models/{name}')

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

    def predict_gesture(self):
        hand_tracking = HandTracking()
        hand_tracking.fetch_hand_landmarks(model=self.model)

if __name__ == '__main__':
    TRAIN = 1
    if TRAIN:
        net = GestureNet(TRAIN)
        net.load_data()
        model = net.create_model()
        net.train_model(model)
    else:
        net = GestureNet(TRAIN)
        net.predict_gesture()
