import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm 

class AutoWork : 
    def __init__(self, dataset_directory, image_size = 150) : 
        self.dataset_directory = dataset_directory
        self.image_size = image_size

        # fetch all of the data 
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        self.label_to_class_name = {}

        # get the input training data 
        train_directory = self.dataset_directory + "/Training"
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
        test_directory = self.dataset_directory + "/Testing"
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

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels
    def train(epochs = 1) : 
        # train function create the model architecture


