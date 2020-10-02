import time
import cv2
import os
import json
import numpy as np

base_path = './save_images/'
image_dir = os.listdir(base_path)


data = {}
data['dataset']=[]
with open('./data.txt', 'w') as outfile:
    json.dump(data, outfile, indent=4)


new_data = {'file_path': 'a',
'bounding_box': 0}



for path in image_dir:

	full_image_path = os.path.join(base_path, path)
	frame = cv2.imread( full_image_path)
	box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

	(x, y, w, h) = [int(v) for v in box]
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		
			
	with open('./data.txt') as json_file:
	    data = json.load(json_file)

	data_stats = data['dataset']

	new_data['file_path'] = full_image_path
	new_data['bounding_box'] = [x,y,w,h]

	data_stats.append(new_data)

	with open('./data.txt', 'w') as outfile:
	    json.dump(data, outfile, indent=4)


	# print (x,y,w,h)

	# cv2.imshow('frame', frame)
	# cv2.waitKey(0)



