
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

#####
"""

Notes: 

-Try adding transforms.Randomize to transform to see what it does to accuracy

-Try using 3d convolutional network

"""
#####

#batch size
batch_size = 1
epochs = 20

ReSize = 32
CenterCrop = 32

# Reszie dataset transformer
transform = transforms.Compose([transforms.Resize(ReSize),
								transforms.CenterCrop(CenterCrop),
								transforms.ToTensor()])




# import training set
traindataset = torchvision.datasets.ImageFolder('brainDataSet/Training/',transform=transform)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

testdataset = torchvision.datasets.ImageFolder('brainDataSet/Testing/',transform=transform)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True)

classes = ("giloma_tumor", "meningioma_tumor","no_tumor","pituitary_tumor")

dataiter = iter(trainloader)
images, labels = dataiter.next()

#print("images shape:",images.shape)
#print("trainloader length:",len(trainloader))
#helpers.imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(3)))





# Define CNN net
class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5) #(in_channels, out_channels, Kernal size)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16* 5*5, 120) #relationship to image size here. # (in features, out features)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 4)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


net = Net()



#############
# Optimzer
#############
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)




#############
# Train
#############


print("##################################")
print("   Testing Brain Tumor Dataset    ")
print("      Image Resize:",ReSize)
print("      CenterCrop:",CenterCrop)
print("      Num Epochs:",epochs)
print("      Batch Size:",batch_size)
print("##################################")


start_time = time.time()

for epoch in range(epochs):  # loop over the dataset multiple times
	print("Now running epoch:",epoch+1," ",end="")

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
		if i == 200:    # print every 200 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 200))
			running_loss = 0.0

print('Finished Training, time to completion in mins:',( (time.time() - start_time)/60) )


#############
# Testing
#############
start_time = time.time()

correct = 0
total = 0
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

print('Accuracy of the network on the 10000 test images: %d %%' % (
	100 * correct / total))

print('Finished Testing, time to completion:',(time.time() - start_time))


