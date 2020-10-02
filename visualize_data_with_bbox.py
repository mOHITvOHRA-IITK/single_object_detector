import time
import cv2
import os
import json
import numpy as np

with open('./data.txt') as json_file:
    data = json.load(json_file)

data_stats = data['dataset']


for i in range(len(data_stats)):

	full_image_path = data_stats[i]['file_path']
	frame = cv2.imread( full_image_path)

	box = data_stats[i]['bounding_box']

	(x, y, w, h) = [int(v) for v in box]
	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow('frame', frame)
	cv2.waitKey(0)



