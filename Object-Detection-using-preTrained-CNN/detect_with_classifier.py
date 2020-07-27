from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from detection_helper import sliding_window
from detection_helper import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2


#construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'path to the input image')
ap.add_argument('-s', '--size', type = str, default = '(200, 150)', help = 'ROI size in pixels')
ap.add_argument('-c', '--min-conf', type = float, default = 0.9, help = 'min probability to filter weak detections')
ap.add_argument('-v', '--visualize', type = int, default = -1, help='whether or not show extra visualization for debugging')
args = vars(ap.parse_args())


#initialize veriables for object detection
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args['size'])
INPUT_SIZE = (224, 224)


#loading network weights
print('Loading network ...')
model = ResNet50(weights = 'imagenet', include_top = True)


#load input image, resize it and then grab its dimensions
orig = cv2.imread(args['image'])
orig = imutils.resize(orig, width = WIDTH)
(H, W) = orig.shape[:2]


#initialize image pyramid
pyramid = image_pyramid(orig, scale = PYR_SCALE, minSize = ROI_SIZE)


#initializing two lists, one for ROIs generated from image pyramid and sliding window, and another to store 
#(x, y) coordinates of ROI
rois = []
locs = []


#measuring how much time it takes to loop over image pyramid and sliding window
start = time.time()


#loop over the image pyramid
for image in pyramid:
	#determin sclae factor b/w original image dimensions and current layer of pyramid
	scale = W / float(image.shape[1])


	#for each layer of pyramid loop over sliding window locatons
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
		
		#scale (x, y) coordinates of ROI w.r.t original image dimension
		x = int(x * scale)
		y = int(y * scale)
		w = int(ROI_SIZE[0] * scale)
		h = int(ROI_SIZE[1] * scale)

		#take ROI and preprocess it so we can later classify it using keras/Tensorflow
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		roi = preprocess_input(roi)

		#update list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))


		#check to see if we are visualizing each of sliding windows in pyramid
		if args['visualize'] > 0:

			#clone original image and draw a bounding box surrounding current region
			clone = orig.copy()
			cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

			#show visualization and current ROI
			cv2.imshow('visualization', clone)
			cv2.imshow('ROI', roiOrig)
			cv2.waitKey(0)


#show how long it took to loop over the image pyramid layers and sliding window location
end = time.time()
print('Looping over Pyramid/Window took {:.5f} seconds'.format(end-start))

#conver the ROIs to a numpy array
rois = np.array(rois, dtype = 'float32')


#classify ROIs using ResNet and show time duration of classification
print('Classifying ROIs...')
start = time.time()
preds = model.predict(rois)
end = time.time()
print('Classifying ROIs took {:.5f} seconds'.format(end - start))

#decode the predictions and initializa a dictionary which maps class labels
#to any ROIs with that label
preds = imagenet_utils.decode_predictions(preds, top = 1)
labels = {}
print('Preds')
print(preds[0])

#loop over the predictions
for (i, p) in enumerate(preds):
	#grab prediction information of current ROI
	(imagenetID, label, prob) = p[0]

	#filter out weak detections by ensuring the predicted probability is greater than
	#min probability
	if prob >= args['min_conf']:
		#grab bounding box associated with prediction and convert coordinates
		box = locs[i]

		#grab list of predictions for label and add bounding box and probability to list
		L = labels.get(label, [])
		L.append((box, prob))
		labels[label] = L


#loop over the labels for each of detected objects in image
for label in labels.keys():
	#clone the original image so we can draw on it
	print('Showing results for "{}" '.format(label))
	clone = orig.copy()

	#loop over all bounding boxes for current label
	for (box, prob) in labels[label]:
		#draw bounding box on image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

	#show results before applying non-maxima supression, then clone image again and show 
	#after applying non-maxima supression
	cv2.imshow('Before', clone)
	clone = orig.copy()

	#extract bounding boxes and associated prediction probabilities and apply
	#non-maxima supression
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)

	#loop over all bounding boxes that were kept after applying non-max supression
	for (startX, startY, endX, endY) in boxes:
		#draw bounding box and label on image
		cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
		y = startY-10 if startY-10>10 else startY+10
		cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

		#show output after applying non-maxima supression
		cv2.imshow('After', clone)
		cv2.waitKey(0)







