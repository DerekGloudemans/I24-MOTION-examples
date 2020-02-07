# contains utility functions relevant for image and coordinate transformation

from __future__ import division
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import time
import torch
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import random

def avg_transform_error(orig,trans):
    n_pts = len(orig)
    sum_error = 0
    
    for i in range(0,n_pts):
        x = orig[i,0]
        y = orig[i,0]
        x_true = trans[i,0]
        y_true = trans[i,1]
        error = np.sqrt((x-x_true)**2+(y-y_true)**2)
        sum_error+= error
    return sum_error/n_pts

    
def transform_pt_array(point_array,M):
    """
    Applies 3 x 3  image transformation matrix M to each point stored in the point array
    """
    
    original_shape = point_array.shape
    
    num_points = int(np.size(point_array,0)*np.size(point_array,1)/2)
    # resize array into N x 2 array
    reshaped = point_array.reshape((num_points,2))   
    
    # add third row
    ones = np.ones([num_points,1])
    points3d = np.concatenate((reshaped,ones),1)
    
    # transform points
    tf_points3d = np.transpose(np.matmul(M,np.transpose(points3d)))
    
    # condense to two-dimensional coordinates
    tf_points = np.zeros([num_points,2])
    tf_points[:,0] = tf_points3d[:,0]/tf_points3d[:,2]
    tf_points[:,1] = tf_points3d[:,1]/tf_points3d[:,2]
    
    tf_point_array = tf_points.reshape(original_shape)
    
    return tf_point_array
    
def get_best_transform(x,y):
    """
    given a set of N points in both x and y space, finds the best (lowest avg error)
    transform of 4 points using oppenCV's getPerspectiveTransform
    returns- transformation matrix M
    """
    # test a simple translation
    if False:
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[1,1],[1,2],[2,1],[2,2]])  
        M_correct = np.array([[1,0,1],[0,1,1],[0,0,1]])
        
    x = np.float32(x)
    y = np.float32(y)
    all_idx = [i for i in range(0,len(x))]
    combos = tuple(combinations(all_idx,4))
    min_err = np.inf
    bestM = 0
    for comb in combos:
         M = cv2.getPerspectiveTransform(x[comb,:],y[comb,:])
         xtf = transform_pt_array(x,M)
         err = avg_transform_error(xtf,y)
         if err < min_err:
             min_err = err
             bestM = M
             bestComb = comb
    return bestM

def velocities_from_pts(point_array,in_coords,out_coords, dt = 1/30.0):
    '''
    point_array - a num_frames x (2*num_objects) array where each row 
    corresponds to a frame and each 2 columns to an object
    in_coords - coordinates from camera space
    out_coords - coordinates in real world feet space
    dt - time between frames
    vel_array - a num_frames-1 x num objects array where each row corresponds
    to the speed of objects between two frames and each column to an object
    '''
    
    # Convert to world_feet_space
    cam_pts = np.load(in_coords)
    world_feet_pts = np.load(out_coords)
    
    # transform points
    M = get_best_transform(cam_pts,world_feet_pts)
    tf_points = transform_pt_array(point_array,M)
    
    # initialize velocity array
    vel_array = np.zeros([np.size(tf_points,0)-1,int(np.size(tf_points,1)/2)])
    
    # i iterates rows/frames, j iterates columns/objects
    for i in range(0,len(vel_array)):
        for j in range(0,len(vel_array[0])):
            #calculate speed for entry i,j
            dx = tf_points[i+1,j*2]-tf_points[i,j*2]
            dy = tf_points[i+1,j*2]-tf_points[i,j*2]
            dist = np.sqrt(dx**2 + dy**2)
            vel_array[i,j] = dist / dt
            
    return vel_array


def plot_velocities(vel_array,dt,smooth_width = 21):
    """
    Pyplot line plot of velocities over time for detected objects with optional smoothing
    """
    fps2mph =  0.681818
    
    plt.figure()
    plt.xlabel('time(s)')
    plt.ylabel('speed(mph)')
    plt.ylim([0,60])
    times = [t*dt for t in range(0,len(vel_array))]
    for col in range(0,len(vel_array[0])):
        obj_vels = vel_array[:,col]*fps2mph
        
        # smooth by convolving hamming window
        width = smooth_width
        hamming = np.hamming(width)
        # smooth and normalize
        smooth = np.convolve(obj_vels,hamming)/sum(hamming)
        # remove edges resulting from convolution
        smooth = smooth[width//2:-width//2+1]
        
        # plot
        plt.plot(times,smooth)
