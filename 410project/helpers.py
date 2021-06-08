
# import torch
# import torchvision
# import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# import torch.nn as nn
# import torch.nn.functional as F

import time
# import keyboard


import itertools
import threading
import time
import sys
import math




def text(text_to_print,num_of_dots,num_of_loops):
    from time import sleep
    import keyboard
    import sys
    shell = sys.stdout.shell
    shell.write(text_to_print,'stdout')
    dotes = int(num_of_dots) * '.'
    for last in range(0,num_of_loops):
        for dot in dotes:
            keyboard.write('.')
            sleep(0.1)
        for dot in dotes:
            keyboard.write('\x08')
            sleep(0.1)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




def calulateConv(w, k, p, s):
	"""
		returns wout

	"""
	res = ((w-k)+(2*p))/s
	return res + 1

def calculatePool(w,k,s):
	return math.floor( ((w-k)/s) +1 )



#print(calulateConv(432,5,0,1))

#print(calculatePool(calulateConv(432,5,0,1),2,2))

