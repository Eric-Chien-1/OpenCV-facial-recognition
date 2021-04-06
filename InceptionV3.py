import os 
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image

DIR_= 'Merge_Data'
train_DIR= os.path.join(DIR_,'Train')
validation_DIR= os.path.join(DIR_,'validation')

#Original idea from https://www.youtube.com/watch?v=d3DJqucOq4g 
#when looking at how to do face mask detection using camera.
#Instead of using an array of categories we append it using os.path.join josh and I use this idea for our prievious program, but just modifying it.
#Based on how we grab files from opencv 3 and the original haarcascade project.


#Directory with our training Mask pictures
train_mask = os.path.join(train_DIR, 'Mask')
#Directory with our trainining No Mask pictures
train_No_mask = os.path.join(train_DIR, 'No Mask')

#Directory with our validation Mask pictures
valid_mask = os.path.join(train_DIR, 'Mask')
#Directory with our validation No Mask pictures
valid__No_mask = os.path.join(train_DIR, 'No Mask')

#https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b
#https://github.com/tejanirla/image_classification/blob/master/transfer_learning.ipynb
#Tells you how to implement the inception v3 in the model and create the layers.
#Importing Inception model
pre_trained_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights='imagenet')

#Making all the layers non trainable(retrain some of the lower layers to increase performance. Keep in mind that this may lead to overfitting)
for layer in pre_trained_model.layers[:-12]:
    layer.trainable= True
    #print(layer)

for layer in pre_trained_model.layers[-12:]:
    layer.trainable= True
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 1.00):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = "Adam", 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_DIR,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_DIR,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary', 
                                                          target_size = (150, 150))

callbacks = myCallback()

from datetime import datetime
time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

csv_logger = tf.keras.callbacks.CSVLogger('training_' + str(time) +'_.log')
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 10,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks, csv_logger])


model.save("my_model" + str(time))
#Test model
numfile=0

TEST_DIR = "Merge_Data/Test/"

folders = []

for data in os.listdir(TEST_DIR):
        #data_list.append(data)
    folders.append(TEST_DIR + (data) + "/")


"""
sum = 2
data_list  = []
for folder in folders:
    sum -= 1
    for img in os.listdir(folder):
        data_list.append([folder + img, sum])

print(data_list)
correct_pred = 0
for data in data_list:
    img = image.load_img(data[0], target_size=(150,150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0]>0.5:
        print(data[0] + " is a mask")
        label = 1
    else:
        print(data[0] + " not a mask")
        label = 0
    
    if(label == data[1]):
        correct_pred += 1
        
accuracy  = "Accuracy: {0}:".format(correct_pred/len(data_list) * 100)

file_object = open('training_' + str(time) +'_.log' , 'a')
file_object.write(accuracy)
file_object.close()
"""


#https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b
#https://github.com/tejanirla/image_classification/blob/master/transfer_learning.ipynb 
for data in folders:
    path= 'Merge_Data/Test/No Mask/' + data
    img = image.load_img(path, target_size=(150,150))
    numfile+=1

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(data + " is a mask!")
    else:
        print(data + " no mask!")

#true_acc = counter/numfile
#print(counter)
#End of program that uses github:
# https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b
#https://github.com/tejanirla/image_classification/blob/master/transfer_learning.ipynb 