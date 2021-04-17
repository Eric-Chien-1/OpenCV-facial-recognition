import cv2
from keras.models import load_model
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('model-007.model')
face_cap = cv2.VideoCapture(0)
label_= {0: 'Mask', 
         1: 'No Mask'}

color_class ={0:(0,255,0),
              1: (0,255,0)}

while(True):
    
    ret, img = face_cap.read()
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x,y,w,h in faces:
        #resize image to a numpy array
        face_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(face_img,(100,100))
        normalized =resized/255.0
        reshaped = np.reshape(normalized,(1,100,100,1))
        result= model.predict(reshaped)

        #place label 
        label = np.argmax(result, axis=1)[0]
        
        cv2.rectangle(img,(x,y), (x+w, y+h), color_class[label],2)
        cv2.rectangle(img,(x,y), (x+w, y+h), color_class[label],2)
        cv2.putText(img, label_[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
    
    cv2.imshow('Live', img)
    key=cv2.waitKey(1)

    if(key==27): 
        break

cv2.destroyAllWindows()
face_cap.release()
