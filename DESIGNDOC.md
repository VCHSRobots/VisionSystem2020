EPIC ROBOTZ Vision System Design Doc
The goal of this project is to create an accurate and robust method of locating an object based on computer vision

Requirements:
Must consistently localize objects relative to the robot
Must provide distance from and angle of view of objects detected
May use up to four cameras at once to locate the object
Must provide output at least several times per second, preferably more
Must be able to run on a Raspberry Pi or Jetson Nano

Potential Solutions:
TensorFlow with Convolutional Object Detection:
Uses a regular TensorFlow object detection models to locate given objects in the frame.
Pros:
  With the right training data, this can lead to fairly accurate results on one camera.
Cons:
  This can be fairly slow -- accurate models tend to run at 7-9 FPS on a GTX 1060
  Does not take advantage of all parallel processing capability of the Jetson
  Can have a hard time detecting the homography of images -- can only detect two possible orientations at best

TensorFlow with Indicated Features for Orientation:
  Pros:
    As accurate as Convolutional Detection
    Can detect homographies with feature points on objects
  Cons:
    Can be fairly slow (7-9 FPS on 1060)
    Each 'feature' must be hand indicated -- over many images, this can be time consuming
    Has been inaccurate in (fairly primitive) tests

TensorFlow Object Detection with Triangulation:
  Pros:
  Cons:
    Requires multiple cameras

ORB Feature Extractor with OpenCV Orientation Detector:

ORB Feature Extractor with TensorFlow Filtering:
Create a convoluted neural network which takes ORB features as opposed to using convolutional feature detectors, then use the inserted feature data to detect object homography. Then use homography to calculate object orientation.
Conclusion:
