# contains utility functions for detection

from __future__ import division
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import time
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def detect_video(video_file, detector, verbose = True, show = True, save_file = None):
    """
    frame by frame object detection of video_file using specified detector
    object as the detector (must have a detect function)
    if save_file is specified, writes annotated video to this file
    returns all_detections - list of Dx8 numpy arrays, one row for each object 
    """
    
    # open up a videocapture object
    cap = cv2.VideoCapture(video_file)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    # opens VideoWriter object for saving video file if necessary
    if save_file != None:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(save_file,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    #main loop   
    start = time.time()
    frames = 0
    ret = True
    all_detections = []
    
    # get first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            
            # detect frame
            im = frame.copy()
            detections,im_out = detector.detect(frame,verbose = False)
            detections = check_classes(detections.cpu().numpy())
            
            if True: # convert to numpy array
                all_detections.append(detections)
            
            #summary statistics
            frames += 1
            print("Video detection rate: {:5.2f} fps".format( frames / (time.time() - start)))
            
            # get next frame or None
            ret, frame = cap.read()
            
             # save frame to file if necessary
            if save_file != None or show:
                im = plot_detections(detections,im)
            
            if save_file != None:
                out.write(im)
            
            # output frame if necessary
            if show:
                im = cv2.resize(im, (1920, 1080))               
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            if frames > 500:
                break
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    torch.cuda.empty_cache()
 
    print("Detection finished")
    return all_detections

def check_classes(detections):
    """
    Simple results modification to better suit the task of roadway object detection
    Removes all detections that are not person,bicycle,car,motorbike,bus,or truck
    Changes trains to trucks since this is a common mistake
    """
    keep_indices = []
    for i in range(len(detections)):
        if detections[i,-1] in [0,1,2,3,5,7]:
            keep_indices.append(i)
        elif detections[i,-1] == 6: #train
            detections[i,-1] = 7 # convert to truck
            keep_indices.append(i)
    return detections[keep_indices,:]
    
def plot_detections(detections, image):
    for item in detections:
        classes = ["person","bicycle","car","motorbike","NA","bus","train","truck"]
        colors = [(255,255,0),(255,0,0),(50,50,200),(0,255,0),(0,255,255),(255,0,255),(100,255,255),(200,50,50)]
        
        # get corner coords
        c1 = tuple(item[1:3].astype(int))
        c2 = tuple(item[3:5].astype(int))
        
        # get class
        cls = int(item[-1])
        label = "{0}".format(classes[cls])
        color = colors[cls]
        
        #plot bounding box
        cv2.rectangle(image, c1, c2,color, 3)
        
        # plot label
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(image, c1, c2,color, -1)
        cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225,255,255], 1);
    return image 

def remove_duplicates(detections):
    """
    for each item in detections (1 frame), removes all identical objects
    """
    
    reduced_detections = []
    for item in detections:
        new_item = np.unique(item,axis = 0)
        reduced_detections.append(new_item)
    return reduced_detections

