import os
import cv2
import pickle
import numpy as np
from PIL import Image


DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(DIR, "Face")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id= 1
#dictionary that store images.
label_ids = {}
y_labels = []
x_train = []
count = 0

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            #placing labels for our images.
            label = os.path.basename(os.path.dirname(path).replace(" ", "-").lower())
            if not label in label_ids:
                #checks if the image has went through our train
                label_ids[label] = current_id
                current_id = 0
            id_= label_ids[label]
            #print(label_ids)
            #y_labels.append(label) # some number
            #x_train.append(label)# verify the image, and turn into numpy array.
            pil_image = Image.open(path).convert("L") #convert to grayscale
            image_array = np.array(pil_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor= 1.5, minNeighbors=5)
            #creates the roi and store it in an array.
            for(x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                print(x_train)
                


#store in the labels in a pickle label
with open("labels.pickle", 'wb') as f:
   pickle.dump(label_ids, f)
#save our recognizer as a yml file
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")