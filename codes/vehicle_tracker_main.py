#!/usr/bin/env python
"""
1.) This code is built to identify and track a vehicle from an image sequence in
the 2014 Visual Object Tracking Challenge.
2.) Three trackers: Cascade Tracker, ShinTomasi Tracker, Lucas Kanade Tracker 
are implemented and compared with the ground truth data

Credit given to OpenCV for library development
Credit given to Brad Philip and Paul Updike for Harr-Cascade Classifier 
training file 

@author: Ruoyu Tan
PhD Student | Pennsylvania State University | t.ruoyu@mgmail.com
Created on March 22, 2015
"""
import numpy as np
import cv2
from ShinTomasiTracker import ShinTomasiTracker
from CascadeTracker import CascadeTracker
from lkTracker import lkTracker
import os
import math
import matplotlib.pyplot as plt
import time

class TrackerPlanner():
    """Identify and track the vehicle from images 

    Attributes:
        image_list: image matrix from original images
        x_true, y_true: centroid position calculated from ground truth data
        TOTAL_SIZE: total numbers of images
        X_INIT, Y_INIT: vehicle initial position
        x_cascade,y_cascade: centroid position obtaind by CascadeTracker 
                             in each frame
        x_shintomasi,y_shintomasi: centroid position obtaind by 
                                   ShinTomasiTracker in each frame
        x_lk,y_lk: centroid position obtaind by Lucas Kanade Tracker 
                   in each frame
        elapsed_cas,elapsed_shi,elapsed_lk: time cost for each tracker
    """
#
    def __init__(self,IMAGE_SIZE):
        """Initiate TrackerPlanner Class"""
        self.image_list=[[] for i in range(IMAGE_SIZE)]
        self.x_true=np.zeros((IMAGE_SIZE,1))
        self.y_true=np.zeros((IMAGE_SIZE,1))
        self.TOTAL_SIZE=IMAGE_SIZE
        self.X_INIT=0
        self.Y_INIT=0
        self.x_cascade=np.zeros((IMAGE_SIZE,1))
        self.y_cascade=np.zeros((IMAGE_SIZE,1))
        self.x_shintomasi=np.zeros((IMAGE_SIZE,1))
        self.y_shintomasi=np.zeros((IMAGE_SIZE,1))
        self.x_lk=np.zeros((IMAGE_SIZE,1))
        self.y_lk=np.zeros((IMAGE_SIZE,1))
        self.elapsed_cas=0
        self.elapsed_shi=0
        self.elapsed_lk=0
    
    def call_cascade(self,CLASS_PATH,SAVE_PATH):
        """call_cascade function: 
        implement Cascade based Tracker to identify and track vehicle

        Args:
            CLASS_PATH: path for training file of Cascade Tracker
            SAVE_PATH:  path to save tracking image
        """
        i_cascade=CascadeTracker(CLASS_PATH)
        start_cas=time.time()
        marked_image,x_cen,y_cen=i_cascade.track_vehicle(self.image_list,
                                 self.X_INIT,self.Y_INIT)
        self.elapsed_cas=time.time()-start_cas
        self.save_image(SAVE_PATH,marked_image)
        self.x_cascade=x_cen
        self.y_cascade=y_cen

    
    def call_shintomasi(self,SAVE_PATH):
        """call_shintomashi function: 
        implement ShinTomasi based Tracker to identify and track vehicle

        Args:
            SAVE_PATH:  path to save tracking image
        """
        i_shintomasi=ShinTomasiTracker()
        start_shi=time.time()
        marked_image,x_cen,y_cen=i_shintomasi.track_vehicle(self.image_list,
                                 self.X_INIT,self.Y_INIT)
        self.elapsed_shi=time.time()-start_shi
        self.save_image(SAVE_PATH,marked_image)
        self.x_shintomasi=x_cen
        self.y_shintomasi=y_cen
    
    def call_lk(self,SAVE_PATH):
        """call_lk function: 
        implement Lucas Kanade based Tracker to identify and track vehicle

        Args:
            SAVE_PATH:  path to save tracking image
        """
        i_lk=lkTracker()
        start_lk=time.time()
        marked_image,x_cen,y_cen=i_lk.track_vehicle(self.image_list,
                                                    self.X_INIT,self.Y_INIT)
        self.elapsed_lk=time.time()-start_lk                             
        self.save_image(SAVE_PATH,marked_image)
        self.x_lk=x_cen
        self.y_lk=y_cen
    
    def result_plot(self):
        """result_plot function: 
        plot the tracking result for comparison
        """
        err_cas=np.zeros(self.TOTAL_SIZE)
        err_shin=np.zeros(self.TOTAL_SIZE)
        err_lk=np.zeros(self.TOTAL_SIZE)
        for i in range(self.TOTAL_SIZE):  
            #Calculate tracking error in each frame
            err_cas[i]=math.sqrt(pow(self.x_cascade[i]-self.x_true[i],2)+
                                 pow(self.y_cascade[i]-self.y_true[i],2))
            err_shin[i]=math.sqrt(pow(self.x_shintomasi[i]-self.x_true[i],2)+
                                  pow(self.y_shintomasi[i]-self.y_true[i],2))
            err_lk[i]=math.sqrt(pow(self.x_lk[i]-self.x_true[i],2)+
                                pow(self.y_lk[i]-self.y_true[i],2))
        i=np.linspace(0,self.TOTAL_SIZE-1,self.TOTAL_SIZE)                
        p1,=plt.plot(i, err_cas, '.r')
        p2,=plt.plot(i, err_shin, 'b')
        p3,=plt.plot(i, err_lk, 'k')        
        plt.legend([p1,p2,p3],['Cascade Tracker','Shin Tomasi Tracker',
                   'Lucas Kanade Tracker'])
        plt.title('Trackers Performance Evaluation')        
        plt.xlabel('Frame Number')
        plt.ylabel('Error between tracking result and ground truth')
        plt.axis([0, self.TOTAL_SIZE, 0, 30])
        plt.show
        print('Cascade Time Cost/sec',self.elapsed_cas)
        print('Shin Tomasi Time Cost/sec',self.elapsed_shi)
        print('Lucas Kanade Time Cost/sec',self.elapsed_lk)
        
    def save_image(self,SAVE_PATH,save_image):
        """save_image function: 
        save tracking images

        Args:
            SAVE_PATH: Path for saving images
            save_image: Image matrix of tracking images
        """
        for i in range(self.TOTAL_SIZE):           
            path=str(i)
            if len(path)==2:
                path="0"+path
            else:
                if len(path)==1:
                    path="00"+path
            save_path=SAVE_PATH+path+'.jpg'
            cv2.imwrite(save_path,save_image[i])
        
    def import_groundtruth(self,TEST_PATH):
        """import_groundtruth function: 
        import ground truth data to evluate tracking result

        Args:
            TEST_PATH: Path of ground truth
        """
        truth_mat=np.loadtxt(TEST_PATH,delimiter=',')
        for i in range(self.TOTAL_SIZE):
            #calculate centroid position of ground truth in each frame
            self.x_true[i]=truth_mat[i][0]+(truth_mat[i][4]-truth_mat[i][0])/2
            self.y_true[i]=truth_mat[i][3]+(truth_mat[i][1]-truth_mat[i][3])/2
        self. X_INIT=self.x_true[0]
        self. Y_INIT=self.y_true[0]
        
    def import_image(self,IMAGE_PATH,IMAGE_FORMAT):
        """import_image function: 
        import original images

        Args:
            IMAGE_PATH: Path of original images
            IMAGE_FORMAT: The format of images
        """
        file_list=[os.path.join(IMAGE_PATH,f) for f in os.listdir(IMAGE_PATH) 
                   if f.endswith(IMAGE_FORMAT)]
        for i in file_list:
            img_seq=int(i[len(IMAGE_PATH):len(IMAGE_PATH)+8])
            img_val=cv2.imread(i,cv2.CV_LOAD_IMAGE_COLOR)            
            self.image_list[img_seq-1]=img_val
                    
def mainloop():
    # Get workspace path
    CODE_PATH=os.path.dirname(os.path.abspath(__file__))
    WS_PATH=CODE_PATH[:len(CODE_PATH)-5]
    # Define path of original images 
    IMAGE_PATH=WS_PATH+"car/"
    IMAGE_FORMAT=".jpg"
    IMAGE_SIZE=252
    # Define path of groundtruth.txt
    TEST_PATH=WS_PATH+"car/groundtruth.txt"
    # Import original images
    i_planner=TrackerPlanner(IMAGE_SIZE)
    i_planner.import_image(IMAGE_PATH,IMAGE_FORMAT)
    # Import groundtruth to evaluate tracker performance
    i_planner.import_groundtruth(TEST_PATH)
    
    # Use Cascade tracker to track vehicle:
    # Define path of training file for Cascade tracker
    CLASS_PATH=WS_PATH+"lib/cars3.xml"
    CASCADE_PATH=WS_PATH+"result/Cascade/"
    i_planner.call_cascade(CLASS_PATH,CASCADE_PATH)
    
    # Since the images have been marked, reload original images
    i_planner.import_image(IMAGE_PATH,IMAGE_FORMAT)
    # Use ShinTomasi tracker to track vehicle:
    SHINTOMASI_PATH=WS_PATH+"result/ShinTomasi/"
    i_planner.call_shintomasi(SHINTOMASI_PATH)
    
    # Since the images have been marked, reload original images
    i_planner.import_image(IMAGE_PATH,IMAGE_FORMAT)
    # Use ShinTomasi tracker to track vehicle:
    LK_PATH=WS_PATH+"result/lk/"
    i_planner.call_lk(LK_PATH)

    #Compare three trackers' results
    i_planner.result_plot()

# Main node function
if __name__ == '__main__':
    mainloop()