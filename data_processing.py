#https://www.youtube.com/watch?v=d3DJqucOq4g
#based on this video
import cv2
import os
import numpy as np
from keras.utils import np_utils

data_dir = 'data\dataset'
categories = os.listdir(data_dir)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories,labels))

img_size = 100
data= []
target= []

for category in categories:
    folder_path= os.path.join(data_dir, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        im_path = os.path.join(folder_path, img_name)
        img= cv2.imread(im_path)

        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Resize image
        resize = cv2.resize(gray,(img_size,img_size))
        data.append(resize)
        #resize our image to be 100x100 and grayscale it
        target.append(label_dict[category])
        #appending image and the label

#convert image to a array
data = np.array(data)/255.0
#by converting the pixel level to 1 and 0s it allows the 
#training to be easier when comparing.
#convert to 4-dimensional array
data = np.reshape(data,(data.shape[0], img_size,img_size,1))
target = np.array(target)
#shows the categorial label of the neural network
new_target = np_utils.to_categorical(target)
np.save('data', data)
np.save('target', new_target)
