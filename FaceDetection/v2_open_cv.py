#version 2.0 of "openCv"
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('monalisa.jpg')
img = cv2.resize(img, None, fx=0.4, fy=0.4)

#Change the image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#sets the program to detect faces using the face_cascade data.
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#Draws rectangle around the faces.
counter = 0
roi_crop = None

for(x,y,w,h) in faces:
    if counter < 1:
        image = cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
        cv2.putText(image, 'Face', (x, y-10),cv2.FONT_HERSHEY_PLAIN, 0.9,(0,255,0),2)
        counter += 1
        roi_crop = image[y:y+h, x:x+w]

cv2.imshow("ROI", roi_crop)
    
cv2.imshow('img', img)

#once esc is pushed the loop would end.
cv2.waitKey(0)