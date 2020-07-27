## Object Detection Using Pre-Trained CNN

In this code we tried to convert a CNN classifier into an object detector using Keras, Tensorflow, and OpenCv. The CNN we used is ResNet 50 and using image Pyramids, sliding window and non-maxima suppression converted the classifier into object detector. 

This code is recreation of blogpost

https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/

## Directory Structure

  |--detection_helper.py
  
  |--detect_with_classifier.py
  
  |--hummingbird.jpg
  
  |--lawn_mower.jpg
  
  |--stingrey.jpg
  
## Demo

In order to run basic demo, you will need to run the command

$ python detect_with_classifier.py --image stingray.jpg --size "(300, 150)"

There are four command line arguments for this program.

1.  -i    Path to the input image
2.  -s    ROI size in pixels
3.  -c    Minimum probability to filter weak detections
4.  -v    Whether or not show extra visualization for debugging

## Results

The code detects objects in images however it is very slow, bounding boxes are not very accurate and we cannot train it end to end.
