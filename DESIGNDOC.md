EPIC ROBOTZ Vision System Design Doc
Authors: Holiday Pettijohn

Overview:
The goal of this project is to create an accurate and robust method of locating an object based on computer vision

Problem Description:
A robot must infer the location of targets based on a video camera for any one of the following reasons:
  The robot must locate and aim at a target with a projectile
  The robot must position itself relative to a target
  The robot must detect the position of an object relative to itself and infer its own position thereby

Specifications:
Must consistently localize objects relative to the robot
Must provide at least the yaw orientation of the objects detected
May use up to four cameras at once to locate the objects
Must be able to run on a Raspberry Pi 3/4 or Jetson Nano at at least 5 FPS (preferably more)
Stretch Goal - Calculate robot position based on known object locations

Milestones:
Start Date: 7/24/2019
Completed:
7/31/2019: Train functional TensorFlow Object Detection model
8/7/2019: ORB/FLANN Object Detector working
Incomplete:
Decompose an OpenCV Homography into object orientation
  Calibrate a camera with OpenCV
  Detect the correct orientation out of the four returned by Homography decomposition
Find Object Orientation from mathematical shape calculations
  Detect the correct orientation out of the two possible options via triangulation
Calculate robot position based on one or more object orientations

Potential Solutions:
TensorFlow with Convolutional Object Detection:
Uses a regular TensorFlow object detection models to locate given objects in the frame. Will be referred to as Convolutional Detector in this DesignDoc.
Pros:
  With enough training data, this can lead to fairly accurate results on one camera
  Can return two possible object orientations
    Object orientation options can be eliminated if multiple key objects are in the frame/view of other cameras and only one set of options is compatible
  Supports NVidia TensorRT for performance optimization
  Requires only one camera
Cons:
  This may be fairly slow on the Jetson -- accurate models tend to run at 7-9 FPS on a GTX 1060
  Does not take advantage of all parallel processing capability of the Jetson
  Relies heavily on the accuracy of TensorFlow Object Detectors
  Can have a hard time detecting the homography of images -- can only detect two possible orientations at best
Work to Implement: Already working on Windows - Awaiting installation on Jetson or other Vision System hardware. Orientation trigonometry is in progress but not yet solved.

TensorFlow Object Detection with Multiple Cameras:
Uses two or more cameras at different angles running the same TensorFlow detection model to triangulate object location and distinguish between the two possible orientations returned by each camera
Pros:
  Allow objects to be detected by triangulating center point of detection
  Can find the correct object orientation based on the options of different camera perspectives
  Uses Parallel Processing on Jetson
Cons:
  Is same speed or slower than Convolutional Detector
  Requires multiple cameras
  Relies heavily on the accuracy of TensorFlow Object Detectors
  Does not all object orientation will be detected in all directions
  Mathematically complicated and possibly computationally heavy
  Does not require OpenCV Camera Tuning or Homography Decomposition
Work to Implement: Some of the trigonometry for orientation is already solved, but the solution is incomplete. Code to find the correct the object orientations has yet to be written.
  
TensorFlow with Trained Sub-Objects for Orientation:
Pros:
  Theoretically as accurate as Convolutional Detection
  Can detect homographies with feature points on objects
  Requires only one camera
Cons:
  Uses Parallel Processing on Jetson
  Is same speed or slower than Convolutional Detector
  Each 'feature' must be hand indicated over many images - this can be time consuming
  Has been inaccurate in (fairly primitive) tests
Work to Implement: Camera Calibration has yet to be figured out - it may take some time

ORB Feature Extractor with Quality Filtering:
Uses ORB features extracted from multiple training images, and uses a quality filter to pick the best ones. The chosen features are then used in detection to find a Homography and object orientation.
Note: With only one training image, ORB was not reliable in experimental tests. However, multiple images and a good quality filter should solve that issue.
Pros:
  Only uses one camera
  Can decompose Homographies with no additional work aside from calibration
  Training data is easier to handle than with TensorFlow Solutions
Cons:
  May need more than one camera to eliminate extraneous returns from Homography decomposition
Work to Implement: Camera Calibration must be working and software must be written to detect good features. The latter should not be hard to implement and test.

