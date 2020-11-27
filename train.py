import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from PIL import Image
from torchvision import transforms

import AlexNet
import utils

net = AlexNet.Net()
criterion = nn.MSELoss()
# criterion = AlexNet.SIELoss
l_list = []

for epoch in range(10):
	optimizer = optim.SGD(net.parameters(), lr=0.8/(epoch+1))
	l = 0
	for i  in range(1400):
		ppm = Image.open('images/{}.jpg'.format(str(i)))
		pgm = Image.open('depths/{}.jpg'.format(str(i)))
		inp = utils.preprocess(ppm).unsqueeze(0)
		tar = utils.tarprocess(pgm).unsqueeze(0)
		target = torch.as_tensor(tar.view(1,4070),dtype=torch.float32)
		optimizer.zero_grad()
		output = net(inp)
		loss = criterion(output, target)
		l += loss.item()
		print('| epoch:%d | image:%d | loss:%f |.1'%(epoch,i,loss.item()))
		loss.backward()
		optimizer.step()

		inp = utils.preprocess(ppm.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
		tar = utils.tarprocess(pgm.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
		target = torch.as_tensor(tar.view(1,4070),dtype=torch.float32)
		optimizer.zero_grad()
		output = net(inp)
		loss = criterion(output, target)
		l += loss.item()
		print('| epoch:%d | image:%d | loss:%f |.2'%(epoch,i,loss.item()))
		loss.backward()
		optimizer.step()


	print('=============',l/1400,'=============')
	l_list.append(l/1400)
		# params = list(net.parameters())
		

PATH = 'AlexNet-dynamic-strengthen.pth'
torch.save(net.state_dict(), PATH)

print(l_list)
# utils.present(14,net,'')

for j in range(49):
	utils.present(i+j,net,'output-%d.jpg'%j)







