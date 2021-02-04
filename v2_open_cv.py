#version 3.0 of "openCv"
import cv2
import os
import glob
from tkinter import *
from tkinter import filedialog
from PIL import Image

def getPath():

    root = Tk()

    root.foldername = filedialog.askdirectory(title = 'Select where data is')
    folderPath = str(root.foldername)
    
    print(folderPath)

    return folderPath

def run_haarcascade(folderPath):

    #store data into array
    data_list = []
    #roi counter
    roi_file_counter = 1
    #store coordinate of subjects into an array
    rois = []
    roi_crop = None
    #prompt to ask where to store roi
    saveDir = filedialog.askdirectory(title='Where would you like to save the ROI?')

    #reading in data
    for data in os.listdir(folderPath):
        if data.endswith('.jpg') or data.endswith('.jpeg') or data.endswith('.png'):
            data_list.append(data)

    #process data
    for data in data_list:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        img = cv2.imread(folderPath + '/' + data)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)

        #Change the image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #sets the program to detect faces using the face_cascade data.
        faces = face_cascade.detectMultiScale(gray, 1.1, 10) #(image, scale factor, number of neighbors)

        print ("Found {0} faces!".format(len(faces)))

        #Draws rectangle around the faces in file.
        for(x,y,w,h) in faces:
            image = cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.putText(image, 'Face', (x, y-10),cv2.FONT_HERSHEY_PLAIN, 0.9,(0,255,0),2)

            roi_crop = image[y:y+h, x:x+w]
            rois.append(roi_crop)

        #cycles through orignal pictures 
        print("Press ESC to cycle through face(s).")
        cv2.imshow("Original Picture", img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

    #cycles through region of interest extracted from origina pictures
    print("Press ESC to cycle through ROI.")
    for i in rois:
        cv2.imshow("ROI", i)
        cv2.imwrite(os.path.join(saveDir, "ROI"+str(roi_file_counter)+".png"),i)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
        roi_file_counter +=1

run_haarcascade(getPath())