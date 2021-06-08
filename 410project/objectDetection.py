import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from torch.utils import data
import cv2
import os
import argparse
import random
import time
import cv2
import torchvision.models as models
from PIL import Image
import urllib
import urllib.request
import matplotlib.pyplot as plt

#############
# Basic Object Detection
#############
image_pathgg1 = "/Users/nicholastitzler/Library/Mobile Documents/com~apple~CloudDocs/Documents/2020:2021College/last term!/410/410project/brainDataSet/Training/glioma_tumor/gg (1).jpg"
image_pathgg2 = "/Users/nicholastitzler/Library/Mobile Documents/com~apple~CloudDocs/Documents/2020:2021College/last term!/410/410project/brainDataSet/Training/glioma_tumor/gg (2).jpg"
image_pathgg5 = "/Users/nicholastitzler/Library/Mobile Documents/com~apple~CloudDocs/Documents/2020:2021College/last term!/410/410project/brainDataSet/Training/glioma_tumor/gg (5).jpg"


def detect(image_path):
	image = cv2.imread(image_path)

	# initialize OpenCV's selective search implementation and set the
	# input image
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)

	# use fast search
	ss.switchToSelectiveSearchQuality()

	# run selective search on the input image
	start = time.time()
	rects = ss.process()
	end = time.time()
	# show how along selective search took to run along with the total
	# number of returned region proposals
	print("[INFO] selective search took {:.4f} seconds".format(end - start))
	print("[INFO] {} total region proposals".format(len(rects)))


	for i in range(0, len(rects), 100):
		output = image.copy()  

		for (x,y,w,h) in rects[i:i + 100]:
			color = [random.randint(0,255) for j in range(0,3)]
			#cv2.rectangle(output, pt1, pt2)

			#resize image
			crop_img = image[y:y+h, x:x+w]
			#cv2.imshow("cropped image",crop_img)

			cv2.rectangle(output, (x,y),(x+w, y+h), color, 2)
			
		

	  
	cv2.imshow("output",output)

	cv2.waitKey(0) 
	  
	#closing all open windows 
	cv2.destroyAllWindows() 

#detect(image_pathgg1)
#detect(image_pathgg2)
#detect(image_pathgg5)


#############
# ResNet 101
#############



model = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#input_image = Image.open(image_pathgg1)

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.request.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)


###

# display results

###

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)


plt.imshow(r)



cv2.waitKey(0) 
	  
#closing all open windows 
cv2.destroyAllWindows() 


