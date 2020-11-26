import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 8))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 8, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 74*55),
		)

	def forward(self, x):
		# print('input: ',x.size())
		x = self.features(x)
		# print('features: ',x.size())
		x = self.avgpool(x)
		# print('avgpool: ',x.size())
		x = torch.flatten(x, 1)
		# print('flatten: ',x.size())
		x = self.classifier(x)
		# print('classifier: ',x.size())
		return x


def SIELoss(output,target,lamb=0.5):
	output = F.relu(output,lower)
	# print('output:',output)
	d = torch.log(output/target+1)
	# print('d:',d)
	sq_sum = torch.mean(torch.square(d))
	sum_sq = torch.square(torch.mean(d))
	loss = sq_sum - sum_sq * lamb
	return loss