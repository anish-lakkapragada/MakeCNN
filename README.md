# MakeCNN

More and more people each day want to apply machine learning and AI into their products. As ML expands beyond the traditional fields, those
who do want to jump into the fun may find themselves confused with the technical concepts involved and amount of learning required to successfully 
apply ML to their job. MakeCNN isto make this simple. **All you do is just give data, and we take care of the rest!** 

The way it works is that you have a folder of image data from various classes, and MakeCNN simply extracts that data, trains
a Convolutional Neural Network off of it, and then allows you to use it to make predictions on more data and apply it. That way 
instead of learning what a convolutional neural network is, you can instead apply it ASAP.

# Installation 
To install simply make sure you are using Python3 and do : 
``` shell
pip install mcnn
```

## More details : 

You must have a folder of data with this exact structure for MakeCNN to work (this is pretty standard in the ML world) : 

Folder_name (you can choose this) : 
 - Training (subfolder)
    - class_1 (subfolder) > contains images
    - class_2 (subfolder) > contains images
    - class_3 (subfolder) > contains images
 - Testing (subfolder)
    - class_1 (subfolder) > contains images
    - class_2 (subfolder) > contains images
    - class_3 (subfolder) > contains images
    
where the "Training" and "Testing" directories are for the training and testing data. Also 
note that MakeCNN can handle however many number of classes you want!

To make this easier to understand, if you wanted to use MakeCNN to predict between cats, dogs, and horses 
you should have a folder like this : 

Any_name 
  - Training
    - cat > contains cat images
    - dog > contains dog images
    - human > contains human images
  - Testing 
    - cat > contains cat images
    - dog > contains dog images
    - human > contains human images 

MakeCNN's main API, AutoWork, was designed to be extremely simple. 

## DEMO

```python
from mcnn.make import AutoWork
aw = AutoWork(dataset_directory = "path_to_folder_containing_data") 
```

When you instantiate the AutoWork class, you give it the data (or really the string that has the path to the data), and all of that is processed. To control the image size of the images in your data
when applied by the CNN model (if you so desire), you can change image_size from 150 (default) to whatever else. 
Don't worry - even if your images are not 150 x 150, they will be resized. 

Then to train : 
```python
aw.train(epochs = 20)
```

Here we train our model for 20 epochs. Easy peasy, lemon squeezy.

If we want to use it, by say evaluating it or predicting it, we have some 
more stuff. To evaluate : 

```python
aw.test_evaluate() # this evaluates it on the test data given
```

Then to predict on a single image : 
```python
aw.predict(path_to_image = "give_path_to_image_file")
```

and you will get your prediction. It will either be "cat", "dog", or "human"
for our example above. You can do this for every image in a file by :

```python
aw.predict_directory(path_to_img_directory = "give_path_to_image_folders")
```

where you will instead get a list of predictions. 

Simple, and easy-to-use - MakeCNN is how we make sure EVERYBODY can leverage ML. 