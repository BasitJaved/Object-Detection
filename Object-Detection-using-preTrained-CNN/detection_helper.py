import imutils

def sliding_window(image, step, ws):

	#image	: input image
	#step	: step size for sliding window better use 4 to 8 pixels
	#ws	: window size (width and height of window) 
	#sliding window across image
	for y in range(0, image.shape[0] - ws[1], step): #loop over rows...y-values
		for x in range(0, image.shape[1] - ws[0], step): #loop over columns...x-values
			#yield current window
			yield(x, y, image[y:y+ws[1], x:x+ws[0]])  #ROI, yield is used instead of return


def image_pyramid(image, scale = 1.5, minSize = (224, 224)):

	#yield original image
	yield image

	#keep looping over image pyramid
	while True:

		#compute dimensions of next image in pyramid
		w = int(image.shape[1]/scale)
		print(w)
		image = imutils.resize(image, width = w)

		#if resized image does not meet the supplied minimum size, stop constructing pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		#yield next image in pyramid
		yield image
