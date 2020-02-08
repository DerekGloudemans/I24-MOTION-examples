from __future__ import division
import torch
import numpy as np
import cv2 
import _pickle as pickle

# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

# import utility functions
from util_detect import detect_video, remove_duplicates
from util_track import condense_detections,track_SORT
from util_transform import get_best_transform, transform_pt_array, velocities_from_pts, plot_velocities, transform_obj_list
from util_draw import draw_world,draw_tracks, draw_track_world


if __name__ == "__main__":
    
    savenum = 0 # assign unique num to avoid overwriting as necessary
    show = True
    
    # name in files
    input_file = '/media/worklab/data_HDD/cv_data/video/1-week-test/Camera_16/cam_1_capture_000.avi'
    
    # name out files
    detect_file = 'output_files/detections.avi'
    track_file =  'output_files/tracks.avi'
    world_file =  'output_files/trajectories.avi'
    comb_file =   'output_files/combo.avi'
    
    # loads model unless already loaded
    try:
       net
    except:
        params = {'cfg_file' :'pytorch_yolo_v3/cfg/yolov3.cfg',
                  'wt_file': 'pytorch_yolo_v3/yolov3.weights',
                  'class_file': 'pytorch_yolo_v3/data/coco.names',
                  'pallete_file': 'pytorch_yolo_v3/pallete',
                  'nms_threshold': 0.5,
                  'conf': 0.52,
                  'resolution': 1024,
                  'num_classes': 80}
        
        net = Darknet_Detector(**params)
        print("Model reloaded.")
    
        # tests that net is working correctly
        if False:
            test ='pytorch_yolo_v3/imgs/person.jpg'
            test ='/media/worklab/data_HDD/cv_data/KITTI/Tracking/Tracks/training/image_02/0000/000000.png'
            out = net.detect(test)
            torch.cuda.empty_cache()    
      
    # get detections or load if already made
    try:
        detections = np.load("output_files/detections{}.npy".format(savenum),allow_pickle= True)
    except:
        detections = detect_video(input_file,net,show = True, save_file=detect_file)
        detections = remove_duplicates(detections)
        
        # remove this later
        np.save("output_files/detections{}.npy".format(savenum), detections)
        
    detections = condense_detections(detections,style = "SORT_cls")
    objs, point_array = track_SORT(detections,mod_err = 1, meas_err = 10, state_err = 1000, fsld_max = 25)
    
    # remove this later
    f = open("output_files/objects{}.cpkl".format(savenum),'wb')
    pickle.dump(objs,f)
    f.close()


    draw_tracks(objs,input_file,track_file,show, trail_size = 50)
    
    # get transform for camera to world space and transform object points
    cam_pts = np.load('point_matching/example_cam_points.npy')
    world_pts = np.load('point_matching/example_world_points.npy')
    gps_pts = np.load('point_matching/example_gps_points.npy')
    background_file = 'point_matching/world.png'
    
    M = get_best_transform(cam_pts,world_pts)
    M2 = get_best_transform(cam_pts,gps_pts) 
    objs = transform_obj_list(objs,M,M2)
    
    # plot together
    draw_world(objs,background_file,world_file,show,trail_size = 20,plot_label = True)
    
    #vel_array = velocities_from_pts(point_array,'im_coord_matching/cam_points2.npy','im_coord_matching/world_feet_points.npy')
    #plot_velocities(vel_array,1/30.0)
    