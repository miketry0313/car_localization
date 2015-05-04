#!/usr/bin/env python
"""
This class is built to identify and track a object from an image sequence based
on Lucas Kanade Tracker.

Credit given to OpenCV for library development

@author: Ruoyu Tan
PhD Student | Pennsylvania State University | t.ruoyu@mgmail.com
Created on March 22, 2015
"""
import cv2
import numpy as np

class lkTracker():
    """Identify and track the vehicle by using Lucas Kanade based tracker """

    def __init__(self):
        pass
            
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
        x_cen=np.zeros(len(raw_image))
        y_cen=np.zeros(len(raw_image))
        kmean_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                          10, 1.0)
        lk_params = dict( winSize  = (21, 21), 
                          maxLevel = 2, 
                          criteria = (cv2.TERM_CRITERIA_EPS | 
                          cv2.TERM_CRITERIA_COUNT, 10, 0.001)) 
        # pre-process the first image        
        image_pre=raw_image[0]
        image_pre=cv2.cvtColor(image_pre, cv2.COLOR_BGR2GRAY)
        image_pre=cv2.equalizeHist(image_pre)
        # use ShinTomasiTracker to obtain feature points for the first image        
        corner = cv2.goodFeaturesToTrack(image_pre,200,0.1,1)
        pre_point=[]
        for i in corner:
                x,y=i.ravel()
                dis=pow(x-X_INIT,2)+pow(y-Y_INIT,2)
                if dis<2000:
                    cv2.circle(raw_image[0],(x,y),3,50,-1)
                    pre_point.append((x,y))
        p0 = np.float32([p for p in pre_point]).reshape(-1, 1, 2)    
        ret,label,center=cv2.kmeans(p0,1,kmean_criteria,10,
                                    cv2.KMEANS_RANDOM_CENTERS)
        x_pre,y_pre=center.ravel()
        x_cen[0],y_cen[0]=x_pre,y_pre
        for i in range(1,len(raw_image)):
            image_pos=raw_image[i]
            image_pos=cv2.cvtColor(image_pos, cv2.COLOR_BGR2GRAY)
            image_pos=cv2.equalizeHist(image_pos)
            p0 = np.float32([p for p in pre_point]).reshape(-1, 1, 2)    
            # use Lucas Kanade Tracker to predict the position of
            # current feature points in next frame
            pos_point, st, err = cv2.calcOpticalFlowPyrLK(image_pre, 
                                        image_pos, p0, None, **lk_params)
            pre_point=[]            
            for j in pos_point:
                x,y=j.ravel()
                pre_point.append((x,y))
                cv2.circle(raw_image[i],(x,y),3,50,-1)
            image_pre=image_pos
            # implement kmean method to find current centroid position
            ret,label,center=cv2.kmeans(p0,1,kmean_criteria,10,
                                        cv2.KMEANS_RANDOM_CENTERS)
            x_pre,y_pre=center.ravel()
            x_cen[i],y_cen[i]=x_pre,y_pre
        return(raw_image,x_cen,y_cen)
   
            