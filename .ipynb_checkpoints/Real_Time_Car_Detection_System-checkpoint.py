#!/usr/bin/env python
# coding: utf-8

import cv2
import yaml
import os
from yaml.loader import SafeLoader
import numpy as np

class Yolo_Pred():
    def __init__(self,onnx_model,data_yaml):
        
        with open(data_yaml,mode='r') as f:
            self.data_yaml=yaml.load(f,Loader=SafeLoader)
            
        self.lables=self.data_yaml['names']
        self.nc=self.data_yaml['nc']
        
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    def predictions(self,Image):
        row, col, d=Image.shape
        
        ## fit image into square image
        max_rc=max(row,col)
        input_image=np.zeros((max_rc,max_rc,3),dtype=np.uint8)
        input_image[:row,:col]=Image
        
        ## Get Prediction from square image
        YOLO_WH_INPUT=640
        blob=cv2.dnn.blobFromImage(input_image,1/255,(YOLO_WH_INPUT,YOLO_WH_INPUT),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        pred=self.yolo.forward()
        
        ## flatten the array
        detection=pred[0]
        detection.shape
        
        
        
        confidances=[]
        boxes=[]
        classes=[]
        
        
        img_w,img_h=input_image.shape[:2]
        x_factor=img_w/YOLO_WH_INPUT
        y_factor=img_h/YOLO_WH_INPUT
        
        for i in range(len(detection)):
            row=detection[i]
            confidance=row[4] # confidence of detection an object
            if confidance>0.4:
                class_score=row[5:].max() # maximum probability from 7 objects
                class_id=row[5:].argmax() # get the index position at which max probabilty occur
                if class_score>0.25:
                    cx,cy,w,h=row[0:4]
                    # construct bounding from four values
                    # left, top, width and height
                    left=int((cx-0.5*w)*x_factor)
                    top=int((cy-0.5*h)*y_factor)
                    width=int(w*x_factor)
                    height=int(h*y_factor)
        
                    box=np.array([left,top,width,height])
        
                    boxes.append(box)
                    confidances.append(confidance)
                    classes.append(class_id)
        
        ## clean
        confidances_np=np.array(confidances).tolist()
        boxes_np=np.array(boxes).tolist()
        
        ## Non Maximum Suppression
        
        index=cv2.dnn.NMSBoxes(boxes_np,confidances_np,0.25,0.45)
        if len(index) > 0:
            index = index.flatten()  # Only flatten if we have valid indices
            for ind in index:
                # extract bounding box
                x,y,w,h=boxes_np[ind]
                bb_conf=int(confidances_np[ind]*100)
                classes_id=classes[ind]
                class_name=self.lables[classes_id]
                
                text=f'{class_name}: {bb_conf}%'
                colors=self.choose_color(classes_id)
                cv2.rectangle(Image,(x,y),(x+w,y+h),colors,2)
                cv2.rectangle(Image,(x,y-30),(x+w,y),colors,-1)
                cv2.putText(Image,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
            return Image
        else:
            return Image
        
    def choose_color(self,Id):
            np.random.seed(10)
            color=np.random.randint(100,255,size=(self.nc,3)).tolist()
            return tuple(color[Id])


