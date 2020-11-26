import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((456,608)),
    transforms.ToTensor()
])

tarprocess = transforms.Compose([
    transforms.Resize((55,74)),
    transforms.ToTensor()
])

def show_vector_as_img(vector):
	img = vector.squeeze()
	img = img.resize(55,74)
	img = img.detach().numpy() * 255
	img = img.astype(np.uint8)
	img = Image.fromarray(img)
	img.show()

def save_vector_as_img(vector,filename):
	img = vector.squeeze()
	img = img.resize(55,74)
	img = img.detach().numpy() * 255
	img = img.astype(np.uint8)
	img = Image.fromarray(img)
	img.save('test-data/' + filename)

def present(index,net,note):
	ppm = Image.open('images/{}.jpg'.format(str(index)))
	inp = preprocess(ppm).unsqueeze(0)
	output = net(inp)
	show_vector_as_img(output)
	# save_vector_as_img(output,'output-%s.jpg' % note)

	# pgm = Image.open('depths/{}.jpg'.format(str(index)))
	# tar = tarprocess(pgm).unsqueeze(0)
	# target = torch.as_tensor(tar.view(1,4070),dtype=torch.float32)
	# show_vector_as_img(target)
	# save_vector_as_img(target,'target-%s.jpg' % note)
