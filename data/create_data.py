import os
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
hands = mp.solutions.hands.Hands()

def crop_hand(img):
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            minx = 9999
            maxy = -9999
            maxx = -9999
            miny = 9999
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)

                # need to find max and min for x and y to draw rectangle around landmarks
                if cx > maxx:
                    maxx = cx
                if cx < minx:
                    minx = cx
                if cy > maxy:
                    maxy = cy
                if cy < miny:
                    miny = cy

            # pad the rectangle 20 pixels from landmarks
            x1 = minx - 20
            y1 = maxy + 20
            x2 = maxx + 20
            y2 = miny - 20

            # pad rectangle to make 300x300 square
            if (x2 - x1) > (y1 - y2):
                square = (x2 - x1) + 5
            if (y1 - y2) > (x2 - x1):
                square = (y1 - y2) + 5
            padx = square - (x2 - x1)
            pady = square - (y1 - y2)

            x2 = x2 + padx
            y1 = y1 + pady

            # cv2.rectangle(img,(x1, y1),(x2, y2),(0,255,0),3)
            # mp.solutions.drawing_utils.draw_landmarks(img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            cropped_img = img[y2:y1,x1:x2]

            # plt.imshow(cropped_img)
            # plt.show()

    # cv2.imshow("Image", img)
            cropped_img = cv2.resize(cropped_img, (227,227), interpolation=cv2.INTER_NEAREST)
    return cropped_img

files = os.listdir('raw_data')
x = []
y = []

print('\n[Processing Data...]\n')

for label in files:
    data_folder = os.listdir(f'raw_data/{label}')
    for file_name in data_folder:

        # img1 = tf.keras.utils.load_img(
        #     path = f'raw_data/{label}/{file_name}',
        #     color_mode = "rgb",
        #     target_size = (227,227)
        # )
        
        img_path = f'raw_data/{label}/{file_name}'
        try:
            img_cropped = crop_hand(img_path)
        except:
            continue
        img_arr = tf.keras.preprocessing.image.img_to_array(img_cropped)
        plt.imshow(img_cropped)
        plt.show()
        x.append(img_arr)
        y.append(label)

print('\n[Saving x and y arrays as .npy]\n')

np.save('x_data.npy', x)
np.save('y_data.npy', y)

print('\n[Done!]\n')