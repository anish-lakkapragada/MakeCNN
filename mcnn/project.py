# got dataset here : https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

global_directory = "/Users/anish/Documents/venv/DVHacksIII/brain_mri"
image_size = 150  # size we resize to 

import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from tqdm import tqdm 

train_images = []
train_labels = []

label_to_class_name =  {} # 0  : "cat", 1 : "dog"

# time to get the input training data 
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

# shuffle the data 

num_images = len(train_images)
shuffled_indices = np.random.permutation(num_images)
train_images, train_labels = train_images[shuffled_indices], train_labels[shuffled_indices]

print(train_images[0])
print(label_to_class_name)



