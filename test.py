import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import network
import cv2
import json
import os

from network_data_formation import generate_network_feed, convert_network_outputs_to_bounding_box, visualize_real_predicted_boxes


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

 


total_iterations = 1
batch_size = 1

frame = None

for i in range(total_iterations):

	image_list = []
	box_param_list = []
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

	x_box_tensor = net.forward(img_tensor, selection_mask_box_param_tensor, 0.0)

	loss = mse_loss(x_box_tensor, box_param_tensor)
	print ("loss: ", loss.cpu().detach().numpy())


	x = x
	y = y

	predicted_param = x_box_tensor.cpu().detach().numpy()
	predicted_obj_conf_score = predicted_param[0,0,x,y]
	print ('predicted_obj_conf_score ', predicted_obj_conf_score)


	predicted_param = x_box_tensor.cpu().detach().numpy()
	predicted_box_param = predicted_param[0,1:5,x,y]
	print ('predicted_box_param ', predicted_box_param)

	predicted_box_param_wrt_image_space = convert_network_outputs_to_bounding_box(predicted_param[0,:,:,:])


	selection_param = selection_mask_box_param_tensor.cpu().detach().numpy()
	selection_box_param = selection_param[0,:,x,y]
	print ('selection_box_param ', selection_box_param)


	real_param = box_param_tensor.cpu().detach().numpy()
	real_obj_conf_score = real_param[0,0,x,y]
	print ('real_obj_conf_score ', real_obj_conf_score)


	real_param = box_param_tensor.cpu().detach().numpy()
	real_box_param = real_param[0,1:5,x,y]
	print ('real_box_param ', real_box_param)

	real_box_param_wrt_image_space = convert_network_outputs_to_bounding_box(real_param[0,:,:,:])




	resize_image = cv2.resize(frame, (320, 240))
	visualize_real_predicted_boxes(resize_image, real_box_param_wrt_image_space, predicted_box_param_wrt_image_space)








