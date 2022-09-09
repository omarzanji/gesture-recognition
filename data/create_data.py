import os
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json
from PIL import Image

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

    cropped_img = np.asarray(image.crop((x1, y1, x2, y2)))
    # cropped_img = img[y2:y1,x1:x2]
    cropped_img = cv2.resize(cropped_img, (227,227), interpolation=cv2.INTER_NEAREST)
    # plt.imshow(cropped_img)
    # plt.show()

    
    return cropped_img

def process_raw_data():
    files = os.listdir('raw_data')

    # x an y labels for gesture prediction mdoel
    x_gesture = []
    y_gesture = []

    # x and y labels for hand detection
    x_tracker = []
    y_tracker = []
    print('\n[Processing Data...]\n')

    for label in files:
        data_folder = os.listdir(f'raw_data/{label}')
        with open(label+'.json', 'r') as f:
            label_dict = json.load(f)
        print(f'processing {label}')
        for cnt,file_name in enumerate(data_folder):
            if cnt % 100 == 0: print(f'{(cnt/5000)*100}%')
            if cnt==6000: break
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
                    img_cropped = crop_hand(img_path, box_coordinates)
                    img_arr = tf.keras.preprocessing.image.img_to_array(img_cropped)
                    # plt.imshow(img_cropped)
                    # plt.show()
                    x_gesture.append(img_arr)
                    y_gesture.append(label)
                else: # use for hand tracker model
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((227,227), Image.Resampling.NEAREST)
                    box_coordinates = img_boxes[ndx]
                    x_tracker.append(tf.keras.preprocessing.image.img_to_array(img))
                    y_tracker.append(box_coordinates)

        
    print('\n[Saving x and y arrays as .npy files]\n')

    np.save('x_gesture_data.npy', x_gesture)
    np.save('y__gesture_data.npy', y_gesture)
    np.save('x_tracker_data.npy', x_gesture)
    np.save('y_tracker_data.npy', y_tracker)

    print('\n[Done!]\n')

process_raw_data()
