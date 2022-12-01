import sqlite3
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import json
import sys
import wget
from zipfile import ZipFile
from PIL import Image
from data.labels import LABELS

def fetch_models(path):
    if not os.path.exists(path+'models/'):
        print('Downloading models.zip ...')
        wget.download('https://github.com/omarzanji/gesture-recognition/releases/download/GestureNet/models.zip')
        with ZipFile('models.zip', 'r') as modelszip:
            modelszip.extractall()
    else:
        print('\nFound models/ folder, loading models...\n')


class HandTracker:
    def __init__(self, train=False, mode=None):
        self.train = train
        self.mode = mode
        if mode == 'prod':
            self.path = 'modules/gesture-recognition/'
        else: self.path = ''
        if train:
            try:
                self.x = np.load(self.path+'data/x_tracker_data.npy')
                self.y = np.load(self.path+'data/y_tracker_data.npy')
                self.x_train = self.x
                self.y_train = self.y
            except:
                print('cached x and y arrays not found...')
        else:
            # self.x = np.load('data/x_tracker_data.npy')
            # self.y = np.load('data/y_tracker_data.npy')
            # self.x_train = self.x
            # self.y_train = self.y
            self.model = keras.models.load_model(self.path+'models/HandTracker')

    def create_model(self):
        # AlexNet 
        model = Sequential([
            # layers.Normalization(),
            layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,1)),
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
        self.hist = model.fit(xtrain, ytrain, epochs=90, verbose=1, batch_size=32)
        loss = self.hist.history['loss']
        acc = self.hist.history['accuracy']
        hist_dict = {"loss": loss, "accuracy": acc}
        self.plot_history(self.hist)
        model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = ypreds
        accuracy = r2_score(ytest, self.ypreds)
        # print(ytest[0:10], self.ypreds[0:10])
        print(f'\n\n[Accuracy: {accuracy}]\n\n')
        self.ytest = ytest
        model.save(f'models/{name}')
        with open(f'{name}_training_loss.json','w') as f: # save training loss data
            json.dump(hist_dict, f)

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

    def get_metrics(self):
        xtrain, xtest, ytrain, ytest = train_test_split(self.x_train, self.y_train, train_size=0.9)
        ypreds = self.model.predict(xtest)
        accuracy = r2_score(ytest, ypreds)
        print(f'Accuracy: {accuracy}')
        self.ypreds = ypreds
        self.ytest = ytest

    def log_gesture(self, gesture):
        '''
        Logs predicted gesture to gestures table in ditto.db sqlite3 database.
        '''
        if not gesture=='no_gesture':
            SQL = sqlite3.connect("ditto.db")
            cur = SQL.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS gestures(gesture VARCHAR)")
            SQL.commit()
            cur.execute("INSERT INTO gestures VALUES('%s')" % gesture)
            SQL.commit()
            SQL.close()

    def predict_hand_tracker(self, gesture_model=None):
        try:
            capture = cv2.VideoCapture(0)
            while True:
                cropped_img = []
                success, img = capture.read()
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(img, cv2.IMREAD_COLOR)
                rgb_img = cv2.resize(rgb, (227,227), cv2.INTER_NEAREST)
                gray_img = cv2.resize(gray, (227,227), cv2.INTER_NEAREST)

                pred = self.model.predict(np.expand_dims(gray_img, 0), verbose=0)
                x1,y1,x2,y2 = pred[0].astype(int)
                img_pil = Image.fromarray(gray_img)
                img_pil_rgb = Image.fromarray(rgb_img.astype('uint8'), 'RGB')
                cropped_img = img_pil.crop((x1,y1,x2,y2))
                cropped_img = np.array(cropped_img) 
                cropped_img_rgb = img_pil_rgb.crop((x1,y1,x2,y2))
                cropped_img_rgb = np.array(cropped_img_rgb) 
                cropped_img_window = cv2.resize(cropped_img_rgb, (300,300), cv2.INTER_NEAREST)
                cropped_img_model = cv2.resize(cropped_img, (227,227), cv2.INTER_NEAREST)
                if gesture_model==None:
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.imshow("Image", rgb)
                    cv2.imshow("Cropped Image", cropped_img_window)
                    cv2.waitKey(300)
                else:
                    gesture_pred = gesture_model.predict(np.expand_dims(cropped_img_model, 0), verbose=0)
                    label = LABELS[np.argmax(np.round(gesture_pred))]
                    if self.mode == 'dev' or self.mode == 'reinforce':
                        cv2.putText(cropped_img_window, str(label), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                        cv2.imshow("Image", rgb)
                        cv2.imshow("Cropped Image", cropped_img_window)
                        if self.mode == 'reinforce':
                            # target = 'no_gesture'
                            target = 'like'
                            if not label == target:
                                if not os.path.exists(f'data/reinforcement/{target}/'):
                                    os.mkdir(f'data/reinforcement/{target}/')
                                filecount = len(os.listdir(f'data/reinforcement/{target}/'))
                                cv2.imwrite(f'data/reinforcement/{target}/{filecount}.png', cropped_img_model)
                        cv2.waitKey(30)
                    elif self.mode == 'prod':
                        self.log_gesture(label)
                        # print(label)
                        cv2.waitKey(500)

        except KeyboardInterrupt:
            capture.release()
            cv2.destroyAllWindows()

    def hand_tracker(self):
        train = self.train
        if train:
            model = self.create_model()
            self.train_model(model)
        else:
            # self.get_metrics()
            self.predict_hand_tracker(test=True)
    

class GestureNet:

    def __init__(self, train=False, mode=None, train_reinforce=None):
        self.mode = mode
        self.train_reinforce = train_reinforce
        
        if mode == 'prod':
            self.path = 'modules/gesture-recognition/'
        else: self.path = ''
        fetch_models(self.path)
        self.labels = LABELS
        if self.train_reinforce: 
            self.train = True
        else: self.train = train
        if self.train:
            try:
                if train_reinforce:
                    self.model = keras.models.load_model(self.path+'models/GestureNet')
                    self.load_reinforcement_data()
                else:
                    self.x = np.load(self.path+'data/x_gesture_data.npy')
                    self.y = np.load(self.path+'data/y_gesture_data.npy')
            except:
                print('cached x and y arrays not found...')
        else:
            self.model = keras.models.load_model(self.path+'models/GestureNet-FineTune')
    

    def load_reinforcement_data(self):
        print('Loading Reinforcement Data for Fine Tuning...')
        x = []
        y = []
        for r_label in os.listdir('data/reinforcement/'):
            for sample in os.listdir(f'data/reinforcement/{r_label}/'):
                image = Image.open(f'data/reinforcement/{r_label}/'+sample).convert('L')
                sample_x = tf.keras.preprocessing.image.img_to_array(image)
                sample_y = np.zeros(len(self.labels))
                sample_y[self.labels.index(r_label)] = 1
                x.append(sample_x)
                y.append(sample_y)
            self.x = np.array(x)
            self.y = np.array(y)


    def load_data(self):
        self.x_train = self.x
        self.y_train = []
        for label in self.y:
            onehot = [0, 0, 0, 0]
            onehot[self.labels.index(label)] = 1
            self.y_train.append(onehot)
        self.y_train = np.array(self.y_train)

    def create_model(self):

        # AlexNet 
        model = Sequential([
            # layers.Normalization(),
            layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,1)),
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
            layers.Dense(4, activation='softmax'),
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
        return model

    def fine_tune(self):
        print('\n\n[Freezing first 6 layers of GestureNet and Finetuning...]\n')
        for layer in self.model.layers[:6]:
            layer.trainable = False
        name = 'GestureNet-FineTune'
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y, train_size=0.9)
        self.hist = self.model.fit(xtrain, ytrain, epochs=20, verbose=1, batch_size=32)
        self.plot_history(self.hist)
        self.model.save(f'models/{name}')
        plt.show()

    def train_model(self, model):
        name = 'GestureNet'
        xtrain, xtest, ytrain, ytest = train_test_split(self.x_train, self.y_train, train_size=0.9)
        self.hist = model.fit(xtrain, ytrain, epochs=90, verbose=1, batch_size=32)
        loss = self.hist.history['loss']
        acc = self.hist.history['accuracy']
        hist_dict = {"loss": loss, "accuracy": acc}
        self.plot_history(self.hist)
        model.summary()
        ypreds = model.predict(xtest)
        self.ypreds = ypreds
        accuracy = accuracy_score(ytest, np.round(self.ypreds).astype(int))
        print(f'\n\n[Accuracy: {accuracy}]\n\n')
        self.ytest = ytest
        model.save(f'models/{name}')
        with open(f'{name}_training_loss.json','w') as f: # save training loss data
            json.dump(hist_dict, f)

    def plot_history(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Model Training Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training loss'], loc='upper right')

    def predict_gesture(self):
        hand_tracking = HandTracker(train=False, mode=self.mode)
        hand_tracking.predict_hand_tracker(gesture_model=self.model)
        

    def gesture_net(self):
        train = self.train
        if train:
            if self.train_reinforce:
                self.fine_tune()
            else:
                self.load_data()
                model = self.create_model()
                self.train_model(model)
        else:
            self.predict_gesture()


def plot_model_metrics():
    with open('GestureNet_training_loss.json', 'r') as f:
        gnet_hist = json.load(f)
    with open('HandTracker_training_loss.json', 'r') as f:
        hnet_hist = json.load(f)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.title('GestureNet Loss')
    plt.plot(gnet_hist['loss'], color='blue', label='loss')
    plt.subplot(1, 2, 2)
    plt.title('GestureNet Accuracy')
    plt.plot(gnet_hist['accuracy'], color='green', label='accuracy')
    plt.xlabel('Epochs')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.title('HandTracker Loss')
    plt.plot(hnet_hist['loss'], color='blue', label='loss')
    plt.subplot(1, 2, 2)
    plt.title('HandTracker Accuracy')
    plt.plot(hnet_hist['accuracy'], color='green', label='accuracy')
    plt.xlabel('Epochs')
    plt.show()


if __name__ == '__main__':

    mode = 'dev'
    if len( sys.argv ) > 1:
        arg = str(sys.argv[1])
        if 'prod' in arg:
            mode = 'prod'
        elif 'reinforce' in arg:
            mode = 'reinforce'
    print(f'\n\n[GestureNet running in {mode} mode.]\n\n')

    # 0 for full gesture recognition, 1 for just tracker
    NET = 0
    TRAIN = False
    TRAIN_REINFORCE = False
    METRICS = False

    if METRICS: 
        plot_model_metrics()
        exit()

    networks = ['GestureNet', 'HandTracker']
    selected = networks[NET]

    if selected == 'GestureNet':
        gest = GestureNet(train=TRAIN, mode=mode, train_reinforce=TRAIN_REINFORCE)
        gest.gesture_net()

    elif selected == 'HandTracker':
        track = HandTracker(train=TRAIN, mode=mode)
        track.hand_tracker()
