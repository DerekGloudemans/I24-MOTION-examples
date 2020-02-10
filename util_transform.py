# contains utility functions relevant for image and coordinate transformation

from __future__ import division
from itertools import combinations
import json
import numpy as np
import cv2 

   

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
 
def transform_obj_list(object_list,M,M2 = None):
    """
    Applies 3 x 3  image transformation matrix M to each point stored in object's
    point list
    object_list - list of KFObjects
    M - 3 x 3 transformation matrix for camera to world image space
    M2 - None or 3 x 3 transformation matrix for camera to gps coordinate space
    """
    
    for i, obj in enumerate(object_list):
        points = obj.all
        num_points = len(points)
        
        
        # add third row
        ones = np.ones([num_points,1])
        points3d = np.concatenate((points,ones),1)
        points3d = points3d[:,[0,1,4]]
        # transform points
        tf_points3d = np.transpose(np.matmul(M,np.transpose(points3d)))
        
        # condense to two-dimensional coordinates
        tf_points = np.zeros([num_points,2])
        tf_points[:,0] = tf_points3d[:,0]/tf_points3d[:,2]
        tf_points[:,1] = tf_points3d[:,1]/tf_points3d[:,2]
        
        object_list[i].all_world = tf_points
        
        if M2 is not None:
            tf_points3d = np.transpose(np.matmul(M2,np.transpose(points3d)))
        
            # condense to two-dimensional coordinates
            tf_points = np.zeros([num_points,2])
            tf_points[:,0] = tf_points3d[:,0]/tf_points3d[:,2]
            tf_points[:,1] = tf_points3d[:,1]/tf_points3d[:,2]
            
            object_list[i].all_gps = tf_points
        
    return object_list
                
def write_json(object_list, metadata,num_frames, out_file = None):
    """
    """
    classes = ["person","bicycle","car","motorbike","NA","bus","train","truck"]

#    metadata = {
#            "camera_id": camera_id,
#            "start_time":start_time,
#            "num_frames":num_frames,
#            "frame_rate":frame_rate
#            }
    data = {}
    
    for frame_num in range(0,num_frames):
        frame_data = []
        
        # for each object
        for i in range(0,len(object_list)):
            obj = object_list[i]
            
            # see if coordinate will be in range
            if obj.first_frame <= frame_num:
                if obj.first_frame + len(obj.all) > frame_num:  
                    veh_data = {}
                    
                    idx = frame_num - obj.first_frame
                    veh_data["id_num"] = i
                    veh_data["class"] = classes[int(obj.cls)]
                    veh_data["detected"] = obj.tags[idx]
                    veh_data["image_position"] = (obj.all[idx]).tolist()
                    veh_data["world_position"] = (obj.all_world[idx]).tolist()
                    veh_data["gps_position"] = (obj.all_gps[idx]).tolist()
                    
                    frame_data.append(veh_data)
        data[frame_num] = frame_data
            
    all_data = {
            "metadata":metadata,
            "data":data    
            }
    if out_file is not None:
        with open(out_file, 'w') as fp:
            json.dump(all_data, fp)
    return all_data
