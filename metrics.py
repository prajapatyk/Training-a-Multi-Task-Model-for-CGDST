# %%
import tensorflow as tf
from shapely.geometry import Polygon
import cv2
import math

imgSize=384
totClasses=8

# %%
#Input arguemnt is only a single box
#(x,y,h,w,sin(theta)) is received in arguments
def evaluateGrasp(y_true,y_pred):
    y_true=y_true.numpy()
    y_pred=y_pred.numpy()
    listTrue=[tuple([float(y_true[0])*imgSize,float(y_true[1])*imgSize])]
    listTrue.append(tuple([float(y_true[2])*imgSize,float(y_true[3])*imgSize]))
        
    listPred=[tuple([float(y_pred[0])*imgSize,float(y_pred[1])*imgSize])]
    listPred.append(tuple([float(y_pred[2])*imgSize,float(y_pred[3])*imgSize]))
    
    trueAngle=math.asin(y_true[-1])
    predAngle=math.asin(y_pred[-1])
    trueAngle=math.degrees(trueAngle)
    predAngle=math.degrees(predAngle)
    diff=abs(trueAngle-predAngle)

    #alpha of 5D representation is converted to theta
    flg1=1
    flg2=1
    if trueAngle<0:
        flg1=-1
    if predAngle<0:
        flg2=-1
    trueAngle=flg1*90-trueAngle
    predAngle=flg2*90-predAngle
    
    listTrue.append(-trueAngle)
    listPred.append(-predAngle)
    boxTrue=cv2.boxPoints(listTrue)
    boxPred=cv2.boxPoints(listPred)
    
    p1=Polygon(boxTrue)
    p2=Polygon(boxPred)
    
    commonArea=p1.intersection(p2).area
    IOU=commonArea/(p1.area+p2.area-commonArea)
    
    if diff<30 and IOU>0.25:
        return True
    else: 
        return False

# %%
def evaluateClassPrediction(y_true,y_pred):
    index=int(tf.math.argmax(y_pred))
    if index==y_true:
        return True
    else:
        return False

# %%
def Caccuracy(y_true,y_pred):
    #print(" Cacccuracy y_true=",y_true[0,:],"y_pred=",y_pred[0,:])
    total=y_true.shape[0]
    if total is None:
        return tf.constant(0)
    Cacc=0
    for i in range(0,total):
        if evaluateClassPrediction(int(y_true[i][0]),y_pred[i][:totClasses]):
            Cacc+=1
    Cacc/=total
    return Cacc


def Laccuracy(y_true,y_pred):
    #print(" Laccuracy y_true=",y_true[0,:],"y_pred=",y_pred[0,:])
    total=y_true.shape[0]
    if total is None:
        return tf.constant(0)
    Lacc=0
    for i in range(0,total):
        isCorrectGrasp=evaluateGrasp(y_true[i][1:],y_pred[i][totClasses:])
        if isCorrectGrasp:
            Lacc+=1
    Lacc/=total
    return Lacc


def Daccuracy(y_true,y_pred):
    #print(" Daccuracy y_true=",y_true[0,:],"y_pred=",y_pred[0,:])
    total=y_true.shape[0]
    if total is None:
        return tf.constant(0)
    Dacc=0
    for i in range(0,total):
        isCorrectClass=evaluateClassPrediction(int(y_true[i][0]),y_pred[i][:totClasses])
        if not isCorrectClass:
            continue
        isCorrectGrasp=evaluateGrasp(y_true[i][1:],y_pred[i][totClasses:])
        if isCorrectGrasp:
            Dacc+=1
    
    Dacc/=total
    return Dacc
