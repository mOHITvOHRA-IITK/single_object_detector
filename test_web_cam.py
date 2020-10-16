import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import network
import cv2
import os

from network_data_formation import generate_network_feed, convert_network_outputs_to_bounding_box, visualize_predicted_boxes






use_GPU = 0




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



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")



while(1):

	timer = cv2.getTickCount()     # start the timer for the calculation of fps in function view_frame
    
	_, img = cap.read()
	real_img = cv2.flip(img, 1) 
	timer = cv2.getTickCount() 

	image_list = []
	img = cv2.resize(real_img, (320, 240))
	img_array = np.array(img, np.float32) / 255.0
	img_array = np.transpose(img_array, (2, 0, 1))
	image_list.append(img_array)

	selection_mask_box_param_list = []
	selection_mask = np.ones(shape=[5, 30, 40], dtype=np.float32)
	selection_mask_box_param_list.append(selection_mask)


	if (use_GPU):
	    img_tensor = torch.from_numpy(np.array(image_list)).cuda()
	    selection_mask_box_param_tensor = torch.from_numpy(np.array(selection_mask_box_param_list)).cuda()
	else:
	    img_tensor = torch.from_numpy(np.array(image_list)).cpu()
	    selection_mask_box_param_tensor = torch.from_numpy(np.array(selection_mask_box_param_list)).cpu()


	x_box_tensor = net.forward(img_tensor, selection_mask_box_param_tensor, 0.0)

	predicted_param = x_box_tensor.cpu().detach().numpy()
	predicted_box_param_wrt_image_space = convert_network_outputs_to_bounding_box(predicted_param[0,:,:,:])
	

	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)      

	visualize_predicted_boxes(real_img, predicted_box_param_wrt_image_space, 2, fps)
	





