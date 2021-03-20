# got dataset here : https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

global_directory = "/Users/anish/Documents/venv/DVHacksIII/brain_mri"
image_size = 150  # size we resize to 
EPOCHS = 30

import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from tqdm import tqdm 

train_images = []
train_labels = []
test_images = []
test_labels = []

label_to_class_name =  {} # 0  : "cat", 1 : "dog"

# get the input training data 
train_directory = global_directory + "/Training"
for index, class_name in enumerate(os.listdir(train_directory)) : 
    class_directory =train_directory + "/" + class_name
    for img in tqdm(os.listdir(class_directory)) : 
        # time to process the image into train_images
        path_to_image = os.path.join(class_directory, img)
        read_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
        image = cv2.resize(read_image, (image_size, image_size))
        train_images.append(image)
        train_labels.append(index)  
    label_to_class_name[index] = class_name

train_images, train_labels = np.array(train_images), np.array(train_labels)


# get testing data 
test_directory = global_directory + "/Testing"
for index, class_name in enumerate(os.listdir(test_directory)) : 
    class_directory =test_directory + "/" + class_name
    for img in tqdm(os.listdir(class_directory)) : 
        # time to process the image into train_images
        path_to_image = os.path.join(class_directory, img)
        read_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
        image = cv2.resize(read_image, (image_size, image_size))
        test_images.append(image)
        test_labels.append(index)  

test_images, test_labels = np.array(test_images), np.array(test_labels)

# shuffle the data 

num_images = len(train_images)
num_classes = len(label_to_class_name)

shuffled_indices = np.random.permutation(num_images)
train_images, train_labels = train_images[shuffled_indices], train_labels[shuffled_indices]
train_images = train_images / 255.0 # also normalize data
test_images = test_images / 255.0

# build model architecture needed 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu, input_shape = (image_size, image_size, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation = tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(256, (3,3), activation = tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(32, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(num_classes, activation = tf.nn.softmax))

if num_classes == 2 : 
    model.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ['accuracy'])
else : 
    model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = EPOCHS, validation_data = (test_images, test_labels))

