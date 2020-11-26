import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,96,11,stride=4)
			nn.ReLU(True)
			F.max_pool2d(2)
			nn.Conv2d(96,256,5,padding=2)
			nn.ReLU(True)
			F.max_pool2d(2)
			nn.Conv2d(256,384,3,padding=1)
			nn.ReLU(True)
			nn.Conv2d(384,384,3,padding=1)
			nn.ReLU(True)
			nn.Conv2d(384,256,3,stride=2)
			nn.ReLU(True)
		)
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 74 * 55),
        )
        
	def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1,self.num_flat_features(x)) # flat for full connect
        x = self.fc(x)
        return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features









