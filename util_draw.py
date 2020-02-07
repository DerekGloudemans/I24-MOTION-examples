# contains utility functions for plotting results of tracking and detections

from __future__ import division
import time
import numpy as np
import cv2 
import random

def draw_tracks(object_list,file_in,file_out = None, show = False, mode = "line",trail_size = 15):
    """
    Takes in list of objects and video file, and plots tracked objects
    objs - list of KF_Objects
    file_in - string, path to video input file
    file_out - string, path to desired video output file
    show - bool
    mode - "line" or "point" - specifies how previous frame locations are displayed
    trail_size - int - specifies num of previous locations to plot
    """

    classes = ["person","bicycle","car","motorbike","NA","bus","train","truck"]
    colors = [(255,255,0),(255,0,0),(50,50,200),(0,255,0),(0,255,255),(255,0,255),(100,255,255),(200,50,50)]

    cap = cv2.VideoCapture(file_in)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(file_in)
    
    # opens VideoWriter object for saving video file if necessary
    if file_out != None:
        # open video_writer object
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))

    start = time.time()
    frame_num = 0
    
    # read first frame
    ret, frame = cap.read()
    
    while cap.isOpened():
        
        if ret:
            
            # for each object
            for i in range(0,len(object_list)):
                obj = object_list[i]
                
                # see if coordinate will be in range
                if obj.first_frame <= frame_num:
                    if obj.first_frame + len(obj.all) > frame_num:  # if there's an error change to strictly greater than
                        bbox = obj.all[frame_num - obj.first_frame]
                        
                        # plot bbox
                        label = "{}:{}".format(classes[int(obj.cls)],i)
                        color = colors[int(obj.cls)]
                        c1 = (int(bbox[0]-bbox[3]*bbox[2]/2),int(bbox[1]-bbox[2]/2))
                        c2 =  (int(bbox[0]+bbox[3]*bbox[2]/2),int(bbox[1]+bbox[2]/2))
                        cv2.rectangle(frame,c1,c2,color,3)
                        
                        # plot label
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,1 , 1)[0]
                        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                        cv2.rectangle(frame, c1, c2,color, -1)
                        cv2.putText(frame, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,1, [225,255,255], 1);
                        
                        # plot previous locations if they were detected at that point
                        if mode == "circle":
                            for j in range(0,trail_size):
                                idx = frame_num - j
                                if idx >= obj.first_frame:
                                    point = obj.all[idx-obj.first_frame]
                                    point = (int(point[0]),int(point[1]))
                                    cv2.circle(frame,point, 4, color, thickness = -1)
                                else:
                                    break
                        # plot previous locations as lines
                        elif mode == "line":
                            prev = (int(bbox[0]),int(bbox[1]))
                            for j in range(0,trail_size,2):
                                idx = frame_num - j
                                if idx >= obj.first_frame:
                                    point = obj.all[idx-obj.first_frame]
                                    point = (int(point[0]),int(point[1]))
                                    cv2.line(frame,point,prev,color, thickness = 3)
                                    prev = point
            # then, try and plot previous locations until reaching trail_size
            
            #summary statistics
            frame_num = frame_num + 1
            print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
            
            # save frame to file if necessary
            if file_out != None:
                out.write(frame)
            
            im = frame.copy()
            # get next frame or None
            ret, frame = cap.read()
            
            # output frame
            if show:
                if frame_width > 1920:
                    im = cv2.resize(im, (int(frame_width/2), int(frame_height/2)))
                else:
                    im = im
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
