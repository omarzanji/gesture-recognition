import os
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json
from PIL import Image

GESTURE = False
TRACK = True
CNT = 5500

hands = mp.solutions.hands.Hands()

def crop_hand(img, box):
    # img = cv2.imread(img, cv2.COLOR_BGR2RGB)
    image = Image.open(img).convert('RGB')
    
    width, height = image.size
    x1, y1, w, h = box
    bbox = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
    int_bbox = np.array(bbox).round().astype(np.int32)

    x1 = int_bbox[0]
    y1 = int_bbox[1]
    x2 = int_bbox[2]
    y2 = int_bbox[3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = h = max(x2 - x1, y2 - y1)
    x1 = max(0, cx - 1.0 * w // 2)
    y1 = max(0, cy - 1.0 * h // 2)
    x2 = cx + 1.0 * w // 2
    y2 = cy + 1.0 * h // 2
    x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))

    crop_box = [x1,y1,x2,y2]
    cropped_img = image.crop((x1, y1, x2, y2))
    cropped_img = np.asarray(cropped_img.resize((227,227), Image.Resampling.NEAREST))
    # cropped_img = img[y2:y1,x1:x2]
    
    plt.imshow(image)
    plt.show()

    
    return image, cropped_img, crop_box

def process_raw_data():
    files = os.listdir('raw_data')

    # x an y labels for gesture prediction mdoel
    x_gesture = []
    y_gesture = []

    # x and y labels for hand detection
    x_tracker = []
    y_tracker = []
    print('\n[Processing Data...]\n')

    data_count = dict()

    for label in files:
        data_count[label] = 0
        data_count['no_gesture'] = 0
        data_folder = os.listdir(f'raw_data/{label}')
        with open(label+'.json', 'r') as f:
            label_dict = json.load(f)
        print(f'processing {label}')
        for cnt,file_name in enumerate(data_folder):
            if cnt % 100 == 0: print(f'{(cnt/CNT)*100}%')
            if cnt==CNT: break
            # img1 = tf.keras.utils.load_img(
            #     path = f'raw_data/{label}/{file_name}',
            #     color_mode = "rgb",
            #     target_size = (227,227)
            # )
            
            img_path = f'raw_data/{label}/{file_name}'
            label_dict_key = file_name.strip('.jpg')

            # img_box per img_label (same size)
            img_boxes = label_dict[label_dict_key]['bboxes']
            img_labels = label_dict[label_dict_key]['labels']
            for ndx,imglabel in enumerate(img_labels):
                if imglabel == label: # if label matches hand with ground truth use for gesture model
                    box_coordinates = img_boxes[ndx]
                    img, img_cropped, crop_box = crop_hand(img_path, box_coordinates)
                    img_cropped_arr = tf.keras.preprocessing.image.img_to_array(img_cropped)
                    img_arr = tf.keras.preprocessing.image.img_to_array(img)
                    # plt.imshow(img)
                    # plt.show()
                    if GESTURE:
                        data_count[label]+=1
                        x_gesture.append(img_cropped_arr)
                        y_gesture.append(label)
                    if TRACK:
                        if not GESTURE: data_count[label] +=1
                        x_tracker.append(img_arr)
                        y_tracker.append(crop_box)
                else: # use for hand tracker model
                    box_coordinates = img_boxes[ndx]
                    img, img_cropped, crop_box = crop_hand(img_path, box_coordinates)
                    # plt.imshow(img_cropped)
                    # plt.show()
                    img_cropped_arr = tf.keras.preprocessing.image.img_to_array(img_cropped)
                    img_arr = tf.keras.preprocessing.image.img_to_array(img)
                    if TRACK:
                        data_count['no_gesture'] += 1
                        # x_tracker.append(img_arr)
                        # y_tracker.append(crop_box)
                    if GESTURE:
                        if not TRACK: data_count['no_gesture'] += 1
                        x_gesture.append(img_cropped_arr)
                        y_gesture.append('no_gesture')

    print(f'\n[Data Counts: {data_count}]')
    print('[Saving x and y arrays as .npy files]\n')

    if GESTURE:
        np.save('x_gesture_data.npy', x_gesture)
        np.save('y_gesture_data.npy', y_gesture)
    if TRACK:
        np.save('x_tracker_data.npy', x_tracker)
        np.save('y_tracker_data.npy', y_tracker)

    print('\n[Done!]\n')


process_raw_data()
