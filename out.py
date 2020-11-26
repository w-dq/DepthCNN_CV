import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from PIL import Image
from torchvision import transforms

import AlexNet
import utils

def save(vector,filename):
	img = vector.squeeze()
	img = img.resize(55,74)
	img = img.detach().numpy() * 255
	img = img.astype(np.uint8)
	img = Image.fromarray(img)
	img = img.resize((640,480))
	img.save('output/' + filename)

net = AlexNet.Net()
PATH = 'pretrain.pth'
net.load_state_dict(torch.load(PATH))

with torch.no_grad():
	for i in range(100):
		ppm = Image.open('images/{}.jpg'.format(str(i)))
		inp = utils.preprocess(ppm).unsqueeze(0)
		output = net(inp)
		save(output,'%s.jpg' % str(i))