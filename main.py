from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mediapipe as mp

from PIL import Image
from data.labels import LABELS

class HandTracker:
    def __init__(self, train=False):
        self.train = train
        if train:
            try:
                self.x = np.load('data/x_tracker_data.npy')
                self.y = np.load('data/y_tracker_data.npy')
                self.x_train = self.x
                self.y_train = self.y
            except:
                print('cached x and y arrays not found...')
        else:
            self.model = keras.models.load_model('models/HandTracker')

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
            layers.Dense(4, activation='relu'),
        ])
        
        model.compile(loss='mse', optimizer='adam', metrics='accuracy')
        return model

    def train_model(self, model):
        name = 'HandTracker'
        xtrain, xtest, ytrain, ytest = train_test_split(self.x_train, self.y_train, train_size=0.9)
        self.hist = model.fit(xtrain, ytrain, epochs=90, verbose=1)
        self.plot_history(self.hist)
        model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = ypreds
        accuracy = r2_score(ytest, self.ypreds)
        print(ytest[0:10], self.ypreds[0:10])
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

    def predict_hand_tracker(self, gesture_model=None):
        try:
            capture = cv2.VideoCapture(0)
            while True:
                cropped_img = []
                success, img = capture.read()
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(rgb, (227,227), cv2.INTER_NEAREST)
                pred = self.model.predict(np.expand_dims(img, 0))
                x1,y1,x2,y2 = pred[0].astype(int)
                img_pil = Image.fromarray(img.astype('uint8'), 'RGB')
                cropped_img = img_pil.crop((x1,y1,x2,y2))
                cropped_img = np.array(cropped_img) 
                # Convert RGB to BGR 
                cropped_img = cropped_img[:, :, ::-1].copy() 
                cropped_img_window = cv2.resize(cropped_img, (300,300), cv2.INTER_NEAREST)
                cropped_img_model = cv2.resize(cropped_img, (227,227), cv2.INTER_NEAREST)
                if gesture_model==None:
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.imshow("Image", rgb)
                    cv2.imshow("Cropped Image", cropped_img_window)
                    cv2.waitKey(1)
                else:
                    gesture_pred = gesture_model.predict(np.expand_dims(cropped_img_model, 0))
                    label = LABELS[np.argmax(np.round(gesture_pred))]
                    cv2.putText(cropped_img_window, str(label), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                    cv2.imshow("Image", rgb)
                    cv2.imshow("Cropped Image", cropped_img_window)
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            capture.release()
            cv2.destroyAllWindows()

    def hand_tracker(self):
        train = self.train
        if train:
            model = self.create_model()
            self.train_model(model)
        else:
            self.predict_hand_tracker(test=True)
    

class GestureNet:

    def __init__(self, train=False):
        self.labels = LABELS
        self.train = train
        if train:
            try:
                self.x = np.load('data/x_gesture_data.npy')
                self.y = np.load('data/y_gesture_data.npy')
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
        hand_tracking = HandTracker(train=False)
        hand_tracking.predict_hand_tracker(gesture_model=self.model)
        

    def gesture_net(self):
        train = self.train
        if train:
            self.load_data()
            model = self.create_model()
            self.train_model(model)
        else:
            self.predict_gesture()

    

if __name__ == '__main__':

    # 0 for full gesture recognition, 1 for just tracker
    NET = 1

    networks = ['GestureNet', 'HandTracker']
    selected = networks[NET]

    if selected == 'GestureNet':
        gest = GestureNet(train=False)
        gest.gesture_net()

    elif selected == 'HandTracker':
        track = HandTracker(train=True)
        track.hand_tracker()
