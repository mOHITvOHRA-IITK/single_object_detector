import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import network
import cv2
import json
import os

from network_data_formation import generate_network_feed


with open('./data.txt') as json_file:
    data = json.load(json_file)

data_stats = data['dataset']









net = network.Net()

gpus = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
net = torch.nn.DataParallel(net).cuda()

weight_path = './weights/net.pth'
if (os.path.isfile(weight_path)):
    net.load_state_dict(torch.load(weight_path))
else:
    print ("no saved weights")
    

net.eval()
torch.no_grad()


mse_loss = nn.MSELoss(reduction = 'sum')
optimizer = optim.Adam(net.parameters(), lr=0.0001)

 


total_iterations = 4000
batch_size = 4

for i in range(total_iterations):

	image_list = []
	box_param_list = []
	label_list = []

	selection_mask_label_list = []
	selection_mask_box_param_list = []

	x = 0
	y = 0

	for j in range(batch_size):

		image_num = np.random.randint(len(data_stats))

		full_image_path = data_stats[image_num]['file_path']
		frame = cv2.imread(full_image_path)
		alpha = np.random.uniform(1.0, 2.0, 1) # Contrast control (1.0-3.0)
		beta = np.random.randint(0, 50) # Brightness control (0-100)
		frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


		box = data_stats[image_num]['bounding_box']



		img_array, object_box_params, y, x, selection_mask_box_param_array = generate_network_feed(frame, box)

		img_array = np.transpose(img_array, (2, 0, 1))
		image_list.append(img_array)

		box_param_list.append(object_box_params)
		selection_mask_box_param_list.append(selection_mask_box_param_array)




	img_tensor = torch.from_numpy(np.array(image_list)).cuda()
	box_param_tensor = torch.from_numpy(np.array(box_param_list)).cuda()
	selection_mask_box_param_tensor = torch.from_numpy(np.array(selection_mask_box_param_list)).cuda()

	x_box_tensor = net.forward(img_tensor, selection_mask_box_param_tensor, 0.5)

	
	
	optimizer.zero_grad()

	loss = mse_loss(x_box_tensor, box_param_tensor)
	

	loss.backward()
	optimizer.step()


	print (i,'/',total_iterations, " with loss: ", loss.cpu().detach().numpy())

torch.save(net.state_dict(), weight_path)


