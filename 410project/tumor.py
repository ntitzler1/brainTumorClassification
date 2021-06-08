
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import time
import helpers

import torch.optim as optim

import cv2
import random

#####
"""

Notes: 

-Try adding transforms.Randomize to transform to see what it does to accuracy

-Try using 3d convolutional network

"""
#####

import itertools
import threading
import time
import sys

done = False
#here is the animation
def animate():
	for c in itertools.cycle(['|', '/', '-', '\\']):
		if done:
			break
		sys.stdout.write('\rloading ' + c)
		sys.stdout.flush()
		time.sleep(0.1)
	sys.stdout.write('\rDone!     ')



#long process here





#batch size
epochs = 6
batch_size = 10
ReSize = 232
CenterCrop = 232
lr = .001

# Reszie dataset transformer
transform = transforms.Compose([transforms.Resize(ReSize),
								transforms.CenterCrop(CenterCrop),
								transforms.ToTensor(),
								transforms.Normalize((.5, .5, .5), (.5, .5, .5))])



# import training set 
traindataset = torchvision.datasets.ImageFolder('brainDataSet/Training/',transform=transform)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = torchvision.datasets.ImageFolder('brainDataSet/Testing/',transform=transform)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True)

classes = ("giloma_tumor", "meningioma_tumor","no_tumor","pituitary_tumor")

#dataiter = iter(testloader)
#images, labels = dataiter.next()






#helpers.imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', '   '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
# print("GroundTruth Lables:",labels)
#exit()


# Define CNN net
class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5) #(in_channels, out_channels, Kernal size)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 55 * 55, 120) #relationship to image size here. # (in features, out features)
		self.fc2 = nn.Linear(120, 120)
		self.fc3 = nn.Linear(120,84)
		self.fc4 = nn.Linear(84, 4)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 55 * 55)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x


net = Net()





#############
# Optimzer
#############
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

print("##################################")
print("   Testing Brain Tumor Dataset    ")
print("      Image Resize:",ReSize)
print("      CenterCrop:",CenterCrop)
print("      Num Epochs:",epochs)
print("      Batch Size:",batch_size)
print("      LR:",lr)
print("##################################")



#############
# Object Detection
#############
# image_path = "/Users/nicholastitzler/Library/Mobile Documents/com~apple~CloudDocs/Documents/2020:2021College/last term!/410/410project/brainDataSet/Training/glioma_tumor/gg (1).jpg"

# image = cv2.imread(image_path)

# # initialize OpenCV's selective search implementation and set the
# # input image
# ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# ss.setBaseImage(image)

# # use fast search
# ss.switchToSelectiveSearchQuality()

# # run selective search on the input image
# start = time.time()
# rects = ss.process()
# end = time.time()
# # show how along selective search took to run along with the total
# # number of returned region proposals
# print("[INFO] selective search took {:.4f} seconds".format(end - start))
# print("[INFO] {} total region proposals".format(len(rects)))


# for i in range(0, len(rects), 100):
# 	output = image.copy()  

# 	for (x,y,w,h) in rects[i:i + 100]:
# 		color = [random.randint(0,255) for j in range(0,3)]
# 		#cv2.rectangle(output, pt1, pt2)

# 		#resize image
# 		crop_img = image[y:y+h, x:x+w]
# 		cv2.imshow("cropped image",crop_img)

# 		cv2.rectangle(output, (x,y),(x+w, y+h), color, 2)
# 		break
	

  
# cv2.imshow("output",output)






#############
# Train
#############
start_time = time.time()
t = threading.Thread(target=animate)
t.start()
trainingLoss = []

for epoch in range(epochs):  # loop over the dataset multiple times
	print("Now running epoch:",epoch+1)

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i == 100:    # print every 500 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 100))
			trainingLoss.append(running_loss)
			running_loss = 0.0


print("\n")
done = True
print('Finished Training, time to completion in mins:',( (time.time() - start_time)/60) )


#############
# Testing
#############
start_time = time.time()

correct = 0
total = 0

gilomaacc = 0
milacc = 0
no_tumoracc = 0
pituiaryacc = 0


gtotal = 111
mtotal = 128
ntotal = 105
ptotal = 99

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
	for data in testloader:
		images, labels = data

		# calculate outputs by running images through the network
		outputs = net(images)
		# the class with the highest energy is what we choose as prediction
		_, predicted = torch.max(outputs.data, 1)

		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		#print("predicted:",predicted)
		#print("labels:",labels)
		for i in range(len(labels)):
			if (labels[i] == 0):
				if (labels[i] == predicted[i]):
					gilomaacc += 1
					
			elif (labels[i] == 1):
				if (labels[i] == predicted[i]):
					milacc += 1
			elif (labels[i] == 2):
				if (labels[i] == predicted[i]):
					no_tumoracc += 1
			elif (labels[i] == 3):
				if (labels[i] == predicted[i]):
					pituiaryacc += 1

print(gilomaacc, milacc, no_tumoracc, pituiaryacc)
print("giloma_tumor acc:",(gilomaacc/gtotal) * 100)
print("meningioma_tumor acc:",(milacc/mtotal) * 100)
print("no_tumor acc:",(no_tumoracc/ntotal) * 100)
print("pituitary_tumor acc:",(pituiaryacc/ptotal) * 100)
print("training loss per 100 mini batches:",trainingLoss)

print("\n")

print('\nAccuracy of the network on the 10000 test images: %d %%' % (
	100 * correct / total))

print('Finished Testing, time to completion:',(time.time() - start_time))








