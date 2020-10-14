import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.p1 = 0.5

		self.conv1 = nn.Conv2d(3, 8, 3, stride = 1, padding=1)
		self.bn1 = nn.BatchNorm2d(8)
		self.conv2 = nn.Conv2d(8, 12, 3, stride = 1, padding=1)
		self.bn2 = nn.BatchNorm2d(12)
		self.conv3 = nn.Conv2d(12, 16, 3, stride = 2, padding=1)
		self.bn3 = nn.BatchNorm2d(16)
		self.conv3_ = nn.Conv2d(16, 16, 3, stride = 1, padding=1)
		self.bn3_ = nn.BatchNorm2d(16)


		self.conv4 = nn.Conv2d(16, 32, 3, stride = 1, padding=1)
		self.bn4 = nn.BatchNorm2d(32)
		self.conv5 = nn.Conv2d(32, 48, 3, stride = 1, padding=1)
		self.bn5 = nn.BatchNorm2d(48)
		self.conv6 = nn.Conv2d(48, 64, 3, stride = 2, padding=1)
		self.bn6 = nn.BatchNorm2d(64)

		
		self.conv4_b2 = nn.Conv2d(16, 64, 3, stride = 2, padding=1)
		self.bn4_b2 = nn.BatchNorm2d(64)
		

		self.conv4_b3 = nn.Conv2d(16, 32, 3, stride = 1, padding=1)
		self.bn4_b3 = nn.BatchNorm2d(32)
		self.conv5_b3 = nn.Conv2d(32, 64, 3, stride = 2, padding=1)
		self.bn5_b3 = nn.BatchNorm2d(64)


		self.conv4_b4 = nn.Conv2d(192, 128, 3, stride = 1, padding=1)
		self.bn4_b4 = nn.BatchNorm2d(128)
		self.conv5_b4 = nn.Conv2d(128, 64, 3, stride = 1, padding=1)
		self.bn5_b4 = nn.BatchNorm2d(64)
		



		self.conv7 = nn.Conv2d(64, 32, 3, stride = 1, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.conv8 = nn.Conv2d(32, 32, 3, stride = 1, padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.conv9 = nn.Conv2d(32, 16, 3, stride = 2, padding=1)
		self.bn9 = nn.BatchNorm2d(16)
		self.conv9_ = nn.Conv2d(16, 16, 3, stride = 1, padding=1)
		self.bn9_ = nn.BatchNorm2d(16)



		self.conv15 = nn.Conv2d(16, 12, 3, stride = 1, padding=1)
		self.dropout15 = nn.Dropout2d(p=self.p1)
		self.conv16 = nn.Conv2d(12, 8, 3, stride = 1, padding=1)
		self.dropout16 = nn.Dropout2d(p=self.p1)
		self.conv17 = nn.Conv2d(8, 5, 3, stride = 1, padding=1)



	def forward(self, x1, m1, p):

		self.p1 = p


		x1 = self.bn1(F.relu(self.conv1(x1)))
		x1 = self.bn2(F.relu(self.conv2(x1)))
		x1 = self.bn3(F.relu(self.conv3(x1)))
		x1 = self.bn3_(F.relu(self.conv3_(x1)))


		x1_b1 = self.bn4(F.relu(self.conv4(x1)))
		x1_b1 = self.bn5(F.relu(self.conv5(x1_b1)))
		x1_b1 = self.bn6(F.relu(self.conv6(x1_b1)))

		x1_b2 = self.bn4_b2(F.relu(self.conv4_b2(x1)))


		x1_b3 = self.bn4_b3(F.relu(self.conv4_b3(x1)))
		x1_b3 = self.bn5_b3(F.relu(self.conv5_b3(x1_b3)))


		x1 = torch.cat([x1_b1, x1_b2, x1_b3], axis=1)


		x1 = self.bn4_b4(F.relu(self.conv4_b4(x1)))
		x1 = self.bn5_b4(F.relu(self.conv5_b4(x1)))
		
		
		
		



		x1 = self.bn7(F.relu(self.conv7(x1)))
		x1 = self.bn8(F.relu(self.conv8(x1)))
		x1 = self.bn9(F.relu(self.conv9(x1)))
		x1 = self.bn9_(F.relu(self.conv9_(x1)))


		x_box = self.dropout15(self.conv15(x1))
		x_box = self.dropout16(self.conv16(x_box))
		x_box = (self.conv17(x_box))
		x_box = m1*x_box



		return x_box