# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import dataAugmentation
IMAGE_SIZE = (384,384)

# %%
#This will convert the dataset to tensors so that the model can be trained.


#Here I assume all the images are that of jpg 
#if there are multiple formats code needs to be updated.
#This function will take input of path of a directory which contains
#images and the corresponding text files. This function will return
#the overall data in tensor format. 
#Two tensors are returned, the first one being the input data 
#and the second one being the label data(target data)

def getRandomVal():
    return 0.4
    
def covert_dataset_to_tensors(path, augmentation : bool = False):
    files=os.listdir(path)
    imgAndLabelData = list()
    for file in files:
        s=file[-3:]
        if s!="jpg":
            continue

        if file[:-3]+"txt" not in files:
            continue
        
        imgPath=os.path.join(path,file)
        txtPath=os.path.join(path,file[:-3]+"txt")
        #Normalizing the image pixel
        NORMimg=np.float32(cv2.imread(imgPath))/255

        txtf=open(txtPath,'r')
        label=list()
        #only single label is expected 
        #for multiple labels change in code is required.
        for line in txtf.readlines():
            label=line.split()
            for i in range(0,len(label)):
                label[i]=float(label[i])
        
        imgAndLabelData.append([NORMimg,label])

        if augmentation:
            augImages = dataAugmentation.get_augmented_images(path, file)
            for augImg in augImages:
                NORMaugImg = np.float32(augImg)/255
                imgAndLabelData.append([NORMaugImg, label])
        
    random.shuffle(imgAndLabelData, getRandomVal)
    inputData=list()
    targetData=list()
    for img, label in imgAndLabelData:
        inputData.append(img)
        targetData.append(label)

    inpDataTensor=tf.convert_to_tensor(inputData)
    tarDataTensor=tf.convert_to_tensor(targetData)
    return inpDataTensor,tarDataTensor


