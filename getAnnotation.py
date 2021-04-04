import logging
import argparse
import xml.etree.cElementTree as ET
import sys
import os

def actualFaces():
    
    #path to test folder
    '''
    path_xml = "/Desktop/Code/Python/UndergradProject/getAnnotations/annotations"
    path_img = "/Desktop/Code/Python/UndergradProject/getAnnotations/images"
    '''

    #path to dataset
    path_xml = "/Desktop/Code/Python/UndergradProject/Datasets/FacesWithAndWithoutMask/annotations"
    path_img = "/Desktop/Code/Python/UndergradProject/Datasets/FacesWithAndWithoutMask/images"

    imgs = []
    xmls = []
    counter = 0
    coord = []

    f = open("myfile.txt", "w")

    #stores xmls file in array
    for file_xml in os.listdir(path_xml):
        if file_xml.endswith('.xml'):
            xmls.append(file_xml)

    #store iamges in array
    for file_img in os.listdir(path_img):
        if file_img.endswith(".png"):
            imgs.append(file_img)

    #loops through xml
    counter = 0
    for data in xmls:
        
        #dependincies
        tree = ET.ElementTree(file=os.path.join(path_xml,data))
        root = tree.getroot()

        #write image path
        f.write(path_img +"/"+ imgs[counter] + " ")

        #get boundingbox
        for elem in root.findall('./object'):
            coord.append(elem.find("bndbox/xmin").text)
            f.write(elem.find("bndbox/xmin").text + ",")
            coord.append(elem.find("bndbox/ymin").text)
            f.write(elem.find("bndbox/ymin").text+ ",")
            coord.append(elem.find("bndbox/xmax").text)
            f.write(elem.find("bndbox/xmax").text+ ",")
            coord.append(elem.find("bndbox/ymax").text)
            f.write(elem.find("bndbox/ymax").text+ ",")

            #if mask = 1, no mask = 0
            if(elem.find("name").text == "with_mask"):
                coord.append(1)
                f.write("1"+ " ")
            elif(elem.find("name").text == "without_mask"):
                coord.append(0)
                f.write("0"+ " ")

        counter +=1 
        f.write("\n")
    f.close()

actualFaces()