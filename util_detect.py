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
            detections,im_out = detector.detect(frame, show = False, verbose = False)
            if True: # convert to numpy array
                all_detections.append(detections.cpu().numpy())
            
            #summary statistics
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
            # get next frame or None
            ret, frame = cap.read()
            
             # save frame to file if necessary
            if save_file != None:
                out.write(im_out)
            
            # output frame if necessary
            if show:
                im = cv2.resize(im_out, (1920, 1080))               
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
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


def detect_frames(directory, detector, verbose = True, show = True, save_file = None):
    """
    frame by frame object detection of series of video frames using specified detector
    object as the detector (must have a detect function)
    if save_file is specified, writes annotated video to this file
    returns all_detections - list of Dx8 numpy arrays, one row for each object 
    """
    
    im_list = [os.path.join(directory,item) for item in os.listdir(directory)]
    im_list.sort()
    im = Image.open(im_list[0])
    # opens VideoWriter object for saving video file if necessary
    if save_file != None:
        # open video_writer object
        out = cv2.VideoWriter(save_file,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, im.size)
    
    #main loop   
    start = time.time()
    frames = 0
    all_detections = []
    
    for file in im_list:
            
        # detect frame
        detections,im_out = detector.detect(file, show = False, verbose = False)
        if True: # convert to numpy array
            all_detections.append(detections.cpu().numpy())
        
        #summary statistics
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        
        
         # save frame to file if necessary
        if save_file != None:
            out.write(im_out)
        
        # output frame if necessary
        if show:
#            im = cv2.resize(im_out, (1920, 1080))               
            cv2.imshow("frame", im_out)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
            
    # close all resources used      
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    torch.cuda.empty_cache()
 
    print("Detection finished")
    return all_detections


def remove_duplicates(detections):
    """
    for each item in detections (1 frame), removes all identical objects
    """
    
    reduced_detections = []
    for item in detections:
        new_item = np.unique(item,axis = 0)
        reduced_detections.append(new_item)
    return reduced_detections

def condense_detections(detections,pt_location = "center"):
    """
    input - list of Dx8 numpy arrays corresponding to detections
    pt_location - specifies where the point will be placed in the original bounding box
    idx (always 0 in this imp.), 4 corner coordinates, objectness , score of class with max conf,class idx.
    output - list of D x 2 numpy arrays with x,y center coordinates
    """
    new_list = []
    if pt_location == "center":
        for item in detections:
            coords = np.zeros([len(item),2])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0
                coords[i,1] = (item[i,2]+item[i,4])/2.0
            new_list.append(coords)    
    elif pt_location == "bottom_center":   
        for item in detections:
            coords = np.zeros([len(item),2])
            for i in range(0,len(item)):
                coords[i,0] = (item[i,1]+item[i,3])/2.0
                coords[i,1] = item[i,4]
            new_list.append(coords)          
            
    return new_list
