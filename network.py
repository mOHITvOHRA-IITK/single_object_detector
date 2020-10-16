import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.p1 = 0.5

		self.b1_conv1 = nn.Conv2d(3, 8, 3, stride = 1, padding=1)
		self.b1_bn1 = nn.BatchNorm2d(8)
		self.b1_conv2 = nn.Conv2d(8, 16, 3, stride = 1, padding=1)
		self.b1_bn2 = nn.BatchNorm2d(16)
		self.b1_conv3 = nn.Conv2d(16, 32, 3, stride = 2, padding=1)
		self.b1_bn3 = nn.BatchNorm2d(32)
		self.b1_conv4 = nn.Conv2d(32, 32, 3, stride = 1, padding=1)
		self.b1_bn4 = nn.BatchNorm2d(32)


		self.b2_conv1 = nn.Conv2d(32, 48, 3, stride = 1, padding=1)
		self.b2_bn1 = nn.BatchNorm2d(48)
		self.b2_conv2 = nn.Conv2d(48, 64, 3, stride = 1, padding=1)
		self.b2_bn2 = nn.BatchNorm2d(64)
		self.b2_conv3 = nn.Conv2d(64, 92, 3, stride = 2, padding=1)
		self.b2_bn3 = nn.BatchNorm2d(92)
		self.b2_conv4 = nn.Conv2d(92, 92, 3, stride = 1, padding=1)
		self.b2_bn4 = nn.BatchNorm2d(92)

		
		self.b3_conv1 = nn.Conv2d(92, 108, 3, stride = 1, padding=1)
		self.b3_bn1 = nn.BatchNorm2d(108)
		self.b3_conv2 = nn.Conv2d(108, 120, 3, stride = 1, padding=1)
		self.b3_bn2 = nn.BatchNorm2d(120)
		self.b3_conv3 = nn.Conv2d(120, 144, 3, stride = 2, padding=1)
		self.b3_bn3 = nn.BatchNorm2d(144)
		self.b3_conv4 = nn.Conv2d(144, 144, 3, stride = 1, padding=1)
		self.b3_bn4 = nn.BatchNorm2d(144)


		self.b4_conv1 = nn.Conv2d(144, 128, 3, stride = 1, padding=1)
		self.b4_dp1 = nn.Dropout2d(p=self.p1)
		self.b4_conv2 = nn.Conv2d(128, 64, 3, stride = 1, padding=1)
		self.b4_dp2 = nn.Dropout2d(p=self.p1)
		self.b4_conv3 = nn.Conv2d(64, 32, 3, stride = 1, padding=1)
		self.b4_dp3 = nn.Dropout2d(p=self.p1)
		self.b4_conv4 = nn.Conv2d(32, 5, 3, stride = 1, padding=1)
		





	def forward(self, x1, m1, p):

		self.p1 = p


		x1 = self.b1_bn1(F.relu(self.b1_conv1(x1)))
		x1 = self.b1_bn2(F.relu(self.b1_conv2(x1)))
		x1 = self.b1_bn3(F.relu(self.b1_conv3(x1)))
		x1 = self.b1_bn4(F.relu(self.b1_conv4(x1)))


		x1 = self.b2_bn1(F.relu(self.b2_conv1(x1)))
		x1 = self.b2_bn2(F.relu(self.b2_conv2(x1)))
		x1 = self.b2_bn3(F.relu(self.b2_conv3(x1)))
		x1 = self.b2_bn4(F.relu(self.b2_conv4(x1)))


		x1 = self.b3_bn1(F.relu(self.b3_conv1(x1)))
		x1 = self.b3_bn2(F.relu(self.b3_conv2(x1)))
		x1 = self.b3_bn3(F.relu(self.b3_conv3(x1)))
		x1 = self.b3_bn4(F.relu(self.b3_conv4(x1)))

		

		x1 = self.b4_dp1(self.b4_conv1(x1))
		x1 = self.b4_dp2(self.b4_conv2(x1))
		x1 = self.b4_dp3(self.b4_conv3(x1))
		x1 = self.b4_conv4(x1)
		x_box = m1*x1



		return x_box