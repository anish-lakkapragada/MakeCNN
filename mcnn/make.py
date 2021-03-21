import numpy as np
import os 
import cv2
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AutoWork : 
    """
    AutoWork is the only class you will ever need. Essentially here you just
    give it the path to your dataset, and it will then train a model (CNN) 
    that you can then use to predict on other images. This is useful for people
    who aren't in the ML field but want to apply ML in their work. 

    Make sure your directory is like this : 
    Folder (this name can be whatever)
        Training
            class_1 > images
            class_2 > images
            class_3 > images
        Testing 
            class_1 > images
            class_2 > images
            class_3 > images
    
    For example if you were trying to predict whether something is a cat, a dog, or a human your dataset
    could be : 
    
    Cat_dog_human_dataset 
        Training
            cat > contains cat images
            dog > contains dog images
            human > contains human images
        Testing 
            cat > contains cat images
            dog > contains dog images
            human > contains human images 
    
    where the Training and Testing folders contain the training and testing data images. 
    """

    def __init__(self, dataset_directory, image_size = 150) : 
        """ 
        Parameters : 
        ->> dataset_directory : path to the directory, make sure the directory is in the format stated above
        ->> image_size : how large you want the images to be (optional - 150 default) in your CNN 
            ->> if your images aren't 150, don't worry! We will just resize them to be 150 * 150. 

        """

        # setup the two generators 
        train_datagenerator = ImageDataGenerator(rescale = 1./255)
        val_datagenerator = ImageDataGenerator(rescale = 1./255) # normalize by rescaling
        
        self.num_to_class = {}
        num_classes = 0 
        for dir in os.listdir(dataset_directory + "/Training") :
            if dir.upper() != ".DS_STORE" :
                self.num_to_class[num_classes] = dir # this is to make it more readable for predictions
                num_classes += 1 # count number of classes

        self.class_mode = "binary" if num_classes == 2 else "sparse"

        train_data = train_datagenerator.flow_from_directory(
            directory = dataset_directory + r"/Training/", 
            target_size = (image_size, image_size), 
            batch_size = 25, 
            class_mode = self.class_mode
        )

        val_data =val_datagenerator.flow_from_directory(
            directory = dataset_directory + r"/Testing/", 
            target_size = (image_size, image_size), 
            batch_size = 25, 
            class_mode = self.class_mode
        )

        # save what we will need
        self.train_data, self.val_data = train_data, val_data
        self.image_size = image_size
    def train(self, epochs = 10) : 
        """here we train our model for however many epochs"""

        # first build model architecture
        model = tf.keras.models.Sequential()

        model.add(Conv2D(32,(3,3), input_shape = (self.image_size, self.image_size,3), activation = tf.nn.relu))
        model.add(Conv2D(32, (3,3), activation = tf.nn.relu))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.30))

        model.add(Conv2D(64, (3,3), activation = tf.nn.relu))
        model.add(Conv2D(64, (3,3), activation = tf.nn.relu))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(32, activation=tf.nn.relu))
        model.add(Dense(32, activation=tf.nn.relu))
        # add last layer + compile!
        if self.class_mode == "binary" : 
            model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))
            model.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ['accuracy'])
        
        else : 
            model.add(tf.keras.layers.Dense(num_classes, activation = tf.nn.softmax))
            model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ['accuracy'])

        # time to train
        model.fit(self.train_data, epochs = epochs, validation_data = self.val_data)
        self.model = model # store our model 

    def test_evaluate(self) : 
        # this simply evaluates the model on the test set
        evaluation = self.model.evaluate(self.val_data)
        return evaluation[-1] # this is the one we care about, not the loss
    def predict(self, path_to_image) :
        """get this certain image and predict on it"""

        read_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
        image = cv2.resize(read_image, (self.image_size, self.image_size))
        image = np.expand_dims(image, axis = 0)
        prediction = np.round_(self.model.predict(image))

        # way to get the value for prediction depends on whether
        # the class is binary or categorical 

        if self.class_mode == "binary" : 
            string_prediction = self.num_to_class[int(prediction[0][0])]
            return string_prediction
        else : 
            string_prediction = self.num_to_class[np.argmax(prediction[0])]
            return string_prediction    
    
    def predict_directory(self, path_to_img_directory)   : 
        """ to predict on a set of images"""

        predictions = []
        for img in os.listdir(path_to_img_directory) : 
            # time to process the image into train_images
            path_to_image = os.path.join(path_to_img_directory, img)
            predictions.append(AutoWork.predict(self, path_to_image))
        return predictions
