# I-24 MOTION examples

start with most impressive GIF - 6 poles and map

The I-24 Mobility Technology Interstate Observation Network (MOTION) is a planned open-road testbed that will enable continuous,ongoing coverage of a roadway at the fine-grained vehicle trajectory level. MOTION consists of a network of 400 pole-mounted 4k resolution cameras recording video data that covers a six mile stretch of freeway in its entirety. The raw video data stream ex-ceeds 130 TB/day of traffic data footage that must be processed in real-time to extract precise vehicle locations, trajectories, and other relevant information from the entire monitored portion ofroadway. Data is reported for each of the 180,000 vehicles per day that travel on the roadway throughout the full length of the instrumented freeway. The first phase of MOTION is scheduled for completed construction by the end of 2020 and will consist of a 3-pole, 18 camera deployment covering roughly 1800 feet of roadway. Phase II will consist of the full 6-mile streth of roadway and is scheduled for completion by the end of 2022.

This repository serves as an example of the algorithms that will convert raw video data into global vehicle trajectories. Included are:

- example_trajectories.json - an example of the output trajectories from the processing pipeline
- pipeline.py - 
input file
(optional) output picture
coordinate sets

Object detection - Talk about what we've tried so far.
Picture of detections

Tracking - KF - show math equations for state model and measurement
Picture and gif of tracks

Trajectory extraction - show math and examples



Future work -
extending RCNN with FastTrack
3D object detection - 
Refinements to tracks in 3D space
Refinements to filter method (deal with false positives)
Multiple plane projection (bilinear interpolation)
Eventually using GPS tracks to synch
