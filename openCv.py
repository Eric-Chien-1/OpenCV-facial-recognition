import cv2

#haarcascade pretrain face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture the image that is grabbed from the webcam.
cap = cv2.VideoCapture(0)

while True:
    #Flag,frame = takes in the thing it is reading.
    _,img = cap.read()
    
    #Change the image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #sets the program to detect faces using the face_cascade data.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    #Draws rectangle around the faces.
    for(x,y,w,h) in faces:
        image = cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
        cv2.putText(image, 'Face', (x, y-10),cv2.FONT_HERSHEY_PLAIN, 0.9,(0,255,0),2)
        
    cv2.imshow('img', img)
    
    #once esc is pushed the loop would end.
    k= cv2.waitKey(30) & 0xff
    if k==27:
        break

#Release the video capture object
cap.release()

#Sources:
#https://www.youtube.com/watch?v=7IFhsbfby9s&ab_channel=AdarshMenon
