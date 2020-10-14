import cv2
import numpy as np




def convert_bounding_box_to_network_outputs(box):
	object_box_params = np.zeros(shape=[5, 30, 40], dtype=np.float32)

	(x, y, w, h) = [int(v) for v in box]

	centroid_x = np.int(np.floor(x + w/2))
	centroid_y = np.int(np.floor(y + h/2))

	grid_position_y = np.int(np.floor(centroid_x / 8))
	grid_position_x = np.int(np.floor(centroid_y / 8))

	box_position_offset_x = np.float( (centroid_x - (8*grid_position_y)) / 8 )
	box_position_offset_y = np.float( (centroid_y - (8*grid_position_x)) / 8 )

	box_position_width = np.float( w / 320 )
	box_position_height = np.float( h / 240 )

	object_box_params[:,grid_position_x, grid_position_y] = [1, box_position_offset_x, box_position_offset_y, box_position_width, box_position_height]

	return object_box_params, grid_position_y, grid_position_x





def convert_network_outputs_to_bounding_box(object_box_params):
	

	max_obj_conf_score = object_box_params[0,:,:].max()

	predicted_box = [0, 0, 0, 0]

	if max_obj_conf_score > 0.8:
		indices = np.asarray(np.where(object_box_params[0,:,:] == max_obj_conf_score))
		grid_position_x = indices[0,0]
		grid_position_y = indices[1,0]


		box_position_offset_x = object_box_params[1, grid_position_x, grid_position_y]
		box_position_offset_y = object_box_params[2, grid_position_x, grid_position_y]
		box_position_width = object_box_params[3, grid_position_x, grid_position_y]
		box_position_height = object_box_params[4, grid_position_x, grid_position_y]
		
		w = np.int(np.floor(320 * box_position_width))
		h = np.int(np.floor(240 * box_position_height))

		centroid_x = np.int(8*grid_position_y + np.floor(8*box_position_offset_x))
		centroid_y = np.int(8*grid_position_x + np.floor(8*box_position_offset_y))

		x = np.int(np.floor(centroid_x - w/2.0))
		y = np.int(np.floor(centroid_y - h/2.0))

		predicted_box = [x,y,w,h]


	return predicted_box

	



def visualize_real_predicted_boxes(frame, real_bbox, predicted_bbox):
	(x, y, w, h) = [int(v) for v in real_bbox]
	print ('real', x,y,w,h)
	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

	(x, y, w, h) = [int(v) for v in predicted_bbox]
	print ('predicted', x,y,w,h)
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


	cv2.imshow('visualize_boxes', frame)
	cv2.waitKey(0)







def generate_network_feed(image, box):

	resize_image = cv2.resize(image, (320, 240))
	resize_image = np.array(resize_image, np.float32) / 255.0

	(x, y, w, h) = [int(v/2) for v in box]
	half_box = [x,y,w,h]
	object_box_params, y, x = convert_bounding_box_to_network_outputs(half_box)



	object_box_params_selection_mask = np.zeros(shape=[5, 30, 40], dtype=np.float32)

	x1 = np.random.randint(30)
	y1 = np.random.randint(40)
	object_box_params_selection_mask[:,x1,y1] = [1, 1, 1, 1, 1]

	x1 = np.random.randint(30)
	y1 = np.random.randint(40)
	object_box_params_selection_mask[:,x1,y1] = [1, 1, 1, 1, 1]
	

	object_box_params_selection_mask[:,x,y] = [1, 1, 1, 1, 1]



	return resize_image, object_box_params, y, x, object_box_params_selection_mask






def visualize_predicted_boxes(frame, predicted_bbox):
	
	(x, y, w, h) = [int(v) for v in predicted_bbox]
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

	cv2.imshow('visualize_boxes', frame)
	cv2.waitKey(1)