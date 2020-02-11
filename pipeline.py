from __future__ import division
import torch
import numpy as np
import argparse
import os
# import detector
from pytorch_yolo_v3.yolo_detector import Darknet_Detector

# import utility functions
from util_detect import detect_video, remove_duplicates
from util_track import condense_detections,track_SORT
from util_transform import  transform_obj_list, write_json, get_best_transform
from util_draw import draw_world,draw_tracks


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Get input file, output directory, .')
    parser.add_argument("input",help='<Required> string - input video file path',type = str)
    parser.add_argument("out_dir",help='<Required> string - output file directory',type = str)
    parser.add_argument("--cam",help='string - camera image coordinate numpy file',type = str)
    parser.add_argument("--sat",help='string - satellite image coordinate numpy file',type = str)
    parser.add_argument("--gps",help='string - gps coordinate numpy file',type = str)
    parser.add_argument("--sat_im",help='string - satellite image file path',type = str)

    args = parser.parse_args()
    # parse args
    CONVERT = True
    input_file = args.input
    out_dir = args.out_dir
#    
#    input_file = "test.avi"
#    out_dir = "test"
    try:
        cam_pts = np.load(args.cam)
        world_pts = np.load(args.sat)
        gps_pts = np.load(args.gps)
        background_file = args.sat_im
    except:
        CONVERT = False # only try to convert if conversion coords input
    
    # name out files
    detect_file = os.path.join(out_dir,"detections.avi") 
    track_file = os.path.join(out_dir,"tracks.avi") 
    world_file = os.path.join(out_dir,"trajectories.avi") 
    data_file = os.path.join(out_dir,"data.json")
    
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
      
    # get detections 
    detections,num_frames = detect_video(input_file,net,show = True, save_file=detect_file)
    detections = remove_duplicates(detections)
    detections = condense_detections(detections,style = "SORT_cls")
    
    # get tracks
    objs, _ = track_SORT(detections,mod_err = 1, meas_err = 10, state_err = 1000, fsld_max = 25)
    num_frames = int(num_frames)
    draw_tracks(objs,input_file,track_file,show = True, trail_size = 50,frame_lim = num_frames)
    
    if CONVERT:
        # get transform for camera to world space and transform object points
        M = get_best_transform(cam_pts,world_pts)
        M2 = get_best_transform(cam_pts,gps_pts) 
        objs = transform_obj_list(objs,M,M2)
        
        # plot together
        draw_world(objs,background_file,world_file,show = True,trail_size = 20,plot_label = True,frame_lim = num_frames)
        
        metadata =   {
                "camera_id": "None",
                "start_time":"None",
                "num_frames":num_frames,
                "frame_rate":"Unknown"
                }
        out = write_json(objs,metadata,num_frames,data_file)
    
