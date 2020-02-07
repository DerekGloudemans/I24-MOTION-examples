# contains utility functions for plotting results of tracking and detections

from __future__ import division
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import time
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random


def draw_track(point_array, file_in, file_out = None, show = False, trail_size = 15): 
    """
    Plots point_array on the video_file used to create it
    """
    # load video file 
    cap = cv2.VideoCapture(file_in)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(file_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    # define random colors for each object
    colormaps = [(random.randrange(0,255),random.randrange(0,255), random.randrange(0,255)) for k in range(0,int(len(point_array[0])/2))]
    
    ret = True
    start = time.time()
    frame_num = 0
    
    # read first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            
            # Plot circles
            for i in range(0, int(len(point_array[0])/2)):
                try:
                    center = (int(point_array[frame_num,i*2]),int(point_array[frame_num,(i*2)+1]))
                    cv2.circle(frame,center, 5, colormaps[i], thickness = -1)
                    
                    for j in range(1,trail_size+1):
                        if frame_num - j >= 0:
                            center = (int(point_array[frame_num-j,i*2]),int(point_array[frame_num-j,(i*2)+1]))
                            cv2.circle(frame,center, int(5*0.99**j), colormaps[i], thickness = -1)
                except:
                    pass # last frame is perhaps not done correctly
            im_out = frame #write here
            
            #summary statistics
            frame_num = frame_num + 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
            # save frame to file if necessary
            if file_out != None:
                out.write(im_out)
            
            # get next frame or None
            ret, frame = cap.read()
            
            # output frame
            if show:
                if frame_width > 1920:
                    im = cv2.resize(im_out, (int(frame_width/2), int(frame_height/2)))
                else:
                    im = im_out
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
                
        else:
            break
        
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    
    print("Tracking in camera-space finished.")
    
    
def draw_world(point_array, file_in, file_out = None, show = True):
    """
    outputs a video with points drawn on an image of the the world at each frame's 
    timestep
    """
    
    # load background image 
    world_im = cv2.imread(file_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(world_im.shape[1])
        frame_height = int(world_im.shape[0])
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    # define random colors for each object
    colormaps = [(random.randrange(0,255),random.randrange(0,255), random.randrange(0,255)) for k in range(0,int(len(point_array[0])/2))]
    
    start = time.time()
    frame_num = 0
    # each loop processes one frame
    for fr in point_array:
            # create fresh copy of background
            frame = world_im.copy()
            
            # loop through points_array and plot circles on background
            for i in range(0, int(len(point_array[0])/2)):
                try:
                    center = (int(point_array[frame_num,i*2]),int(point_array[frame_num,(i*2)+1]))
                    cv2.circle(frame,center, 10, colormaps[i], thickness = -1)
                except:
                    pass # last frame is perhaps not done correctly, may also catch points that fall off image boundary
            
            #summary statistics
            frame_num += 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
             # save frame to file if necessary
            if file_out != None:
                out.write(frame)
            
            # output frame
            if show:
                #im = cv2.resize(im_out, (1920, 1080))               
                cv2.imshow("frame", frame)
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


def draw_track_world(point_array,tf_point_array,background_in,video_in,file_out = None, show = True,trail_size = 100):    
    """
    combines draw_track and draw_world into a single output video
    """
    # load video file 
    cap = cv2.VideoCapture(video_in)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_in)
    
    # load background image 
    world_im = cv2.imread(background_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(cap.get(3))+ world_im.shape[1]
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
    # define random colors for each object
    colormaps = [(random.randrange(0,255),random.randrange(0,255), random.randrange(0,255)) for k in range(0,int(len(point_array[0])/2))]
    
    ret = True
    start = time.time()
    frame_num = 0
    
    # get first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            # get background image
            backg = world_im.copy()
            
            for i in range(0, int(len(point_array[0])/2)):
                # draw points on camera frame
                try:
                    center = (int(point_array[frame_num,i*2]),int(point_array[frame_num,(i*2)+1]))
                    cv2.circle(frame,center, 10, colormaps[i], thickness = -1)
                    
                    for j in range(1,trail_size+1):
                        if frame_num - j >= 0:
                            center = (int(point_array[frame_num-j,i*2]),int(point_array[frame_num-j,(i*2)+1]))
                            cv2.circle(frame,center, int(10*0.99**j), colormaps[i], thickness = -1)
                    
                except:
                    pass # last frame is perhaps not done correctly
                    
                # draw points on world frame
                try:
                    center = (int(tf_point_array[frame_num,i*2]),int(tf_point_array[frame_num,(i*2)+1]))
                    cv2.circle(backg,center, 10, colormaps[i], thickness = -1)
                    
                    for j in range(1,trail_size+1):
                        if frame_num - j >= 0:
                            center = (int(tf_point_array[frame_num-j,i*2]),int(tf_point_array[frame_num-j,(i*2)+1]))
                            cv2.circle(backg,center, int(10*0.99**j), colormaps[i], thickness = -1)
                    
                except:
                    pass # last frame is perhaps not done correctly, may also catch points that fall off image boundary
            
            
            # pad backg image
            bottom_pad = frame_height-backg.shape[0]
            pad = cv2.copyMakeBorder(backg, 0 , bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            # combine two images into a single image
            im_out = np.concatenate((frame,pad),axis = 1)
            
            #summary statistics
            frame_num += 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
            # get next frame or None
            ret, frame = cap.read()
            
            # save frame to file if necessary
            if file_out != None:
                out.write(im_out)
            
            # output frame
            if show:
                scale = 0.5
                resize = (int(frame_width * scale),int(frame_height*scale))
                im = cv2.resize(im_out, resize)   
                
                cv2.imshow("frame", im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
        
        else:
            break
        
    # close all resources used      
    cap.release()
    cv2.destroyAllWindows()
    try:
        out.release()
    except:
        pass
    
    print("Combination video writing finished.")
