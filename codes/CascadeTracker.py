#!/usr/bin/env python
"""
This class is built to identify and track a object from an image sequence based
on Cascade Harr Classfier

Credit given to OpenCV for library development

@author: Ruoyu Tan
PhD Student | Pennsylvania State University | t.ruoyu@mgmail.com
Created on March 22, 2015
"""
import cv2
import numpy as np

class CascadeTracker():
    """Identify and track the vehicle by using Cascade based tracker 

    Attributes:
        car_cascade: training file for Cascade Classfier
    """
#
    def __init__(self,CLASS_PATH):
        """Initiates Cascade tracker"""
        self.car_cascade=cv2.CascadeClassifier(CLASS_PATH)       
    
    def track_vehicle(self,input_image,X_INIT,Y_INIT):
        """track_vehicle function: 
        
        Args:
            input_image: original image matrix
            X_INIT, Y_INIT: vehicle initial position 
        
        Returns:
            raw_image: tracking image
            x_cen,y_cen: centroid position obtaind by tracker in each frame
        """
        raw_image=input_image
        rect_count=0
        rect_first=0
        i_cen=np.zeros(len(raw_image))
        x_cen=np.zeros(len(raw_image))
        y_cen=np.zeros(len(raw_image))
        x_pre=np.zeros(len(raw_image))
        y_pre=np.zeros(len(raw_image))
        for i in range(len(raw_image)):
            # pre-process image
            gray=cv2.cvtColor(raw_image[i], cv2.COLOR_BGR2GRAY)
            gray=cv2.equalizeHist(gray)
            # find rect boxes by using cascade classfier
            rects=self.car_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                  minNeighbors=2,minSize=(40, 40), flags=0)
            # select the one rect box which has the closest distance 
            # from previous centroid position            
            min_value=float("inf")
            for (x,y,w,h) in rects:
                if rect_first==0:
                    distance=pow(x+w/2-X_INIT,2)+pow(y+h/2-Y_INIT,2)
                else:
                    distance=pow(x+w/2-x_pre[rect_count],2)+pow(y+h/2-
                             y_pre[rect_count],2)
                if (distance<min_value):
                    rect_first=1
                    min_value=distance
                    (x_min,y_min,w_min,h_min)=(x,y,w,h)
            if (rect_count==0 and rect_first==1) or (min_value<800):  
                 rect_count=rect_count+1
                 i_cen[rect_count]=i
                 x_pre[rect_count]=x_min+w_min/2
                 y_pre[rect_count]=y_min+h_min/2
                 x_cen[i]=x_pre[rect_count]
                 y_cen[i]=y_pre[rect_count]
                 cv2.rectangle(raw_image[i],(x_min,y_min),
                               (x_min+w_min,y_min+h_min),(255,0,0),2)
        return(raw_image,x_cen,y_cen)        
        