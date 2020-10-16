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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', help='Set the number of iteration', type=int, default=1000)
parser.add_argument('-b', '--batch_size', help='Set the batch size', type=int, default=2)
parser.add_argument('-s', '--set_number', help='Set folder number for dataset', type=int, default=1)
args = parser.parse_args()




with open('./save_images/set' + str(args.set_number) + '/data.txt') as json_file:
    data = json.load(json_file)

data_stats = data['dataset']





use_GPU = 1




net = network.Net()


if (use_GPU):
    gpus = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    net = torch.nn.DataParallel(net).cuda()
else:
    device = torch.device('cpu')




weight_path = './weights/net.pth'
if (os.path.isfile(weight_path)):

    if (use_GPU):
        net.load_state_dict(torch.load(weight_path))
    else:
        state_dict = torch.load(weight_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

else:
    print ("no saved weights")
    

net.eval()
torch.no_grad()


mse_loss = nn.MSELoss(reduction = 'sum')
optimizer = optim.Adam(net.parameters(), lr=0.0001)

 


total_iterations = args.iterations
batch_size = args.batch_size

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




	if (use_GPU):
		img_tensor = torch.from_numpy(np.array(image_list)).cuda()
		box_param_tensor = torch.from_numpy(np.array(box_param_list)).cuda()
		selection_mask_box_param_tensor = torch.from_numpy(np.array(selection_mask_box_param_list)).cuda()
	else:
		img_tensor = torch.from_numpy(np.array(image_list)).cpu()
		box_param_tensor = torch.from_numpy(np.array(box_param_list)).cpu()
		selection_mask_box_param_tensor = torch.from_numpy(np.array(selection_mask_box_param_list)).cpu()


	

	x_box_tensor = net.forward(img_tensor, selection_mask_box_param_tensor, 0.5)

	
	
	optimizer.zero_grad()

	loss = mse_loss(x_box_tensor, box_param_tensor)
	

	loss.backward()
	optimizer.step()


	print (i,'/',total_iterations, " with loss: ", loss.cpu().detach().numpy())

torch.save(net.state_dict(), weight_path)


