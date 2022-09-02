import os
import tensorflow as tf
import numpy as np

files = os.listdir('raw_data')
x = []
y = []

print('\n[Processing Data...]\n')

for label in files:
    data_folder = os.listdir(f'raw_data/{label}')
    for file_name in data_folder:

        img1 = tf.keras.utils.load_img(
            path = f'raw_data/{label}/{file_name}',
            color_mode = "rgb",
            target_size = (227,227)
        )
        # img2 = tf.image.adjust_brightness(
        #     img1, 
        # )
        # img2.show()
        # exit()
        img_arr = tf.keras.preprocessing.image.img_to_array(img1)
        x.append(img_arr)
        y.append(label)
        img_arr = tf.keras.preprocessing.image.img_to_array(img1)
        x.append(img_arr)
        y.append(label)

print('\n[Saving x and y arrays as .npy]\n')

np.save('x_data.npy', x)
np.save('y_data.npy', y)

print('\n[Done!]\n')