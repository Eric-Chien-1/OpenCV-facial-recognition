import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

cap = cv2.VideoCapture(0)

while(True):
    #capture each frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Scale factor determine the accuracy same as min Neighbors
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=6)

    for(x, y, w, h) in faces:
        #print cordinates for roi
        print(x, y, w, h)
        #creating the bouding box and only splicing the coordinate around face.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]

        #Using openCV on images with deep learning
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)

        #takes a picture of the roi and storing it a .png file.
        img_item = "my-img.png"
        #only printing coordinates of the bounding box
        cv2.imwrite(img_item, roi_gray)

        #Drawing bounding box
        color = (0,255,0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width,height), color, stroke)


    cv2.imshow('frame', frame)
    #when 'q' is clicked destroy window
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
