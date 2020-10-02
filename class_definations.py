import cv2
import numpy as np
import torch
import posenet
import time




use_cuda = 1

model = posenet.load_model(101)
if (use_cuda):
	model = model.cuda()
else:
	model = model.cpu()
output_stride = model.output_stride
torch.no_grad()




PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]


radius_fraction = 0.06

alpha = 0.4


def write_data3(current_frame, text, rec_x, rec_y, rec_w, rec_h, text_offset_x, text_offset_y, text_format1, text_format2, text_color):

	image_height, image_width, _ = current_frame.shape

	overlay = current_frame.copy()
	cv2.rectangle(current_frame,( np.int(rec_x*image_width), np.int(rec_y*image_height) ), ( np.int((rec_x + rec_w)*image_width), np.int((rec_y + rec_h)*image_height) ), (0,0,0), -1)
	current_frame = cv2.addWeighted(overlay, alpha, current_frame, 1 - alpha, 0)
	cv2.putText(current_frame, str(text),  ( (np.int((rec_x + text_offset_x)*image_width)), (np.int((rec_y + text_offset_y)*image_height)) ) , cv2.FONT_HERSHEY_SIMPLEX, text_format1, text_color, text_format2, cv2.LINE_AA)

	return current_frame


	

class vision_demo_class:

	

	def __init__(self, timer_value):

		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
		    raise IOError("Cannot open webcam")
		    exit()

		# self.cap.set(cv2.CAP_PROP_FPS, 30)
		# self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		# self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

		_, frame = self.cap.read()
		new_h, new_w, layers = frame.shape

		self.fps = 0
		self.keypoints_data = np.zeros(shape=[17,2])
	
		self.image_height = new_h
		self.image_width = new_w

		self.exit_point = np.array([np.int(0.85*self.image_width), np.int(0.15*self.image_height)], dtype=int)
		self.save_point = np.array([np.int(0.85*self.image_width), np.int(0.35*self.image_height)], dtype=int)
		self.timer_point = np.array([np.int(0.85*self.image_width), np.int(0.55*self.image_height)], dtype=int)

		self.exit_button = False
		self.save_button = False
		self.timer_button = False


		self.timer_button_clock_flag = False
		self.timer_button_clock_valve = 0

		self.image_num = 0
		self.action_done = False
		self.timer_value = timer_value
		


	
	
		# if (use_cuda):
		# 	cv2.namedWindow('openpose_demo_GPU', cv2.WND_PROP_FULLSCREEN)
		# 	cv2.setWindowProperty('openpose_demo_GPU',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		# else:
		# 	cv2.namedWindow('openpose_demo_CPU', cv2.WND_PROP_FULLSCREEN)
		# 	cv2.setWindowProperty('openpose_demo_CPU',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



	

	def get_current_frame(self):

		_, frame = self.cap.read()
		frame = cv2.flip(frame, 1) 

		self.timer = cv2.getTickCount()     # start the timer for the calculation of fps in function view_frame
		self.current_frame = frame.copy()  
		self.current_frame2 = frame.copy()  





	def view_frame(self):

	
		self.fps = cv2.getTickFrequency() / (cv2.getTickCount() - self.timer)        # fps calculation with timer start in function get_current_frame.
		self.current_frame = write_data3(self.current_frame, 'fps:' + str(int(self.fps)), 0.05, 0.74, 0.11, 0.10, 0.01, 0.07, 1, 2, (255, 0, 255))
		
		cv2.imshow('openpose_demo_GPU', self.current_frame)
		cv2.waitKey(1)







	def get_human_keypoints(self, scaling_factor):

		frame = self.current_frame2
		height, width, layers = frame.shape
		new_h=np.int(height/scaling_factor)
		new_w=np.int(width/scaling_factor)
		frame = cv2.resize(frame, (new_w, new_h)) 

			
		input_image, draw_image, output_scale = posenet.my_read_imgfile(frame, 1.0, output_stride)


		if (use_cuda):
			input_image = torch.Tensor(input_image).cuda()
			heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
			pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses_GPU(
		            scaling_factor*heatmaps_result.squeeze(0),
		            scaling_factor*offsets_result.squeeze(0),
		            scaling_factor*displacement_fwd_result.squeeze(0),
		            scaling_factor*displacement_bwd_result.squeeze(0),
		            output_stride=scaling_factor*output_stride,
		            max_pose_detections=1,
		            min_pose_score=0.25)
		else:
			input_image = torch.Tensor(input_image).cpu()
			heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
			pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses_GPU(
		            scaling_factor*heatmaps_result.squeeze(0),
		            scaling_factor*offsets_result.squeeze(0),
		            scaling_factor*displacement_fwd_result.squeeze(0),
		            scaling_factor*displacement_bwd_result.squeeze(0),
		            output_stride=scaling_factor*output_stride,
		            max_pose_detections=1,
		            min_pose_score=0.25)


		keypoint_coords *= output_scale
		self.keypoints_data = np.array(keypoint_coords[0]) 



		i = PART_NAMES.index('leftWrist')
		j = PART_NAMES.index('rightWrist')
		p2 = (int(self.keypoints_data[i,1]), int(self.keypoints_data[i,0]) )
		p1 = (int(self.keypoints_data[j,1]), int(self.keypoints_data[j,0]) )


		i = PART_NAMES.index('leftElbow')
		j = PART_NAMES.index('rightElbow')
		p22 = (int(self.keypoints_data[i,1]), int(self.keypoints_data[i,0]) )
		p11 = (int(self.keypoints_data[j,1]), int(self.keypoints_data[j,0]) )

		if (p1[0] > 0 and p1[1] > 0 and p2[0] >0 and p2[1] > 0 
			and p11[0] > 0 and p11[1] > 0 and p22[0] >0 and p22[1] > 0 ):

			left_x_diff = p2[0] - p22[0]
			left_y_diff = p2[1] - p22[1]
			left_dis = np.sqrt(left_x_diff*left_x_diff + left_y_diff*left_y_diff)
			

			right_x_diff = p1[0] - p11[0]
			right_y_diff = p1[1] - p11[1]
			right_dis = np.sqrt(right_x_diff*right_x_diff + right_y_diff*right_y_diff)

			left_palm_x = 0
			left_palm_y = 0

			right_palm_x = 0
			right_palm_y = 0
			

			if (left_dis > 1.0):
				left_x_diff /= left_dis
				left_y_diff /= left_dis
				left_palm_x = np.int(p2[0] + 0.5*left_dis*left_x_diff)
				left_palm_y = np.int(p2[1] + 0.5*left_dis*left_y_diff)
			else:
				left_palm_x = np.int(p2[0])
				left_palm_y = np.int(p2[1])

			

			if (right_dis > 1.0):
				right_x_diff /= right_dis
				right_y_diff /= right_dis
				right_palm_x = np.int(p1[0] + 0.5*right_dis*right_x_diff)
				right_palm_y = np.int(p1[1] + 0.5*right_dis*right_y_diff)
			else:
				right_palm_x = np.int(p1[0])
				right_palm_y = np.int(p1[1])


			
			i = PART_NAMES.index('leftWrist')
			j = PART_NAMES.index('rightWrist')
		
			if (left_palm_x < self.image_width):
				self.keypoints_data[i,1] = left_palm_x

			if (left_palm_y < self.image_height):
				self.keypoints_data[i,0] = left_palm_y

			if (right_palm_x < self.image_width):
				self.keypoints_data[j,1] = right_palm_x

			if (right_palm_y < self.image_height):
				self.keypoints_data[j,0] = right_palm_y



	
	def buttons(self, array, background_color, txt, text_color):

		overlay = self.current_frame.copy()
		cv2.circle(self.current_frame, (array[0], array[1]), np.int(radius_fraction*self.image_width), background_color, -1)
		self.current_frame = cv2.addWeighted(overlay, alpha, self.current_frame, 1 - alpha, 0)
		cv2.putText(self.current_frame, txt,  ( (array[0]- np.int(2*radius_fraction*self.image_width/3)), (array[1] + np.int(radius_fraction*self.image_height/3)) ) , cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3, cv2.LINE_AA)




	def button_selected(self, array):

		i = PART_NAMES.index('leftWrist')
		j = PART_NAMES.index('rightWrist')
		p1 = (int(self.keypoints_data[i,1]), int(self.keypoints_data[i,0]) )
		p2 = (int(self.keypoints_data[j,1]), int(self.keypoints_data[j,0]) )


		if (p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0):

			
			cv2.circle(self.current_frame, p1, np.int(radius_fraction*self.image_width/3), (0, 0, 255), -1)
			cv2.circle(self.current_frame, p2, np.int(radius_fraction*self.image_width/3), (0, 0, 255), -1)

			lw_pt_x = array[0] - p1[0]
			lw_pt_y = array[1] - p1[1]
			dis_lw_pt = np.sqrt(lw_pt_x*lw_pt_x + lw_pt_y*lw_pt_y)

			rw_pt_x = array[0] - p2[0]
			rw_pt_y = array[1] - p2[1]
			dis_rw_pt = np.sqrt(rw_pt_x*rw_pt_x + rw_pt_y*rw_pt_y)

			if (dis_lw_pt < np.int(radius_fraction*self.image_width) or dis_rw_pt < np.int(radius_fraction*self.image_width)):
				return True
			else:
				return False
		
		return False




	def button_left(self, array):

		i = PART_NAMES.index('leftWrist')
		j = PART_NAMES.index('rightWrist')
		p1 = (int(self.keypoints_data[i,1]), int(self.keypoints_data[i,0]) )
		p2 = (int(self.keypoints_data[j,1]), int(self.keypoints_data[j,0]) )


		if (p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0):

			
			cv2.circle(self.current_frame, p1, np.int(radius_fraction*self.image_width/3), (0, 0, 255), -1)
			cv2.circle(self.current_frame, p2, np.int(radius_fraction*self.image_width/3), (0, 0, 255), -1)

			lw_pt_x = array[0] - p1[0]
			lw_pt_y = array[1] - p1[1]
			dis_lw_pt = np.sqrt(lw_pt_x*lw_pt_x + lw_pt_y*lw_pt_y)

			rw_pt_x = array[0] - p2[0]
			rw_pt_y = array[1] - p2[1]
			dis_rw_pt = np.sqrt(rw_pt_x*rw_pt_x + rw_pt_y*rw_pt_y)

			if (dis_lw_pt > np.int(3*radius_fraction*self.image_width) and dis_rw_pt > np.int(3*radius_fraction*self.image_width)):
				return True
			else:
				return False
		
		return False
				





	def contact_less_new_GUI(self):

		self.get_human_keypoints(1)
		
		
		if (self.exit_button == False):
			self.buttons(self.exit_point, (0, 0,255), 'Ext', (255, 255, 255))
			self.exit_button = self.button_selected(self.exit_point)
		else:
			print()
			print ("********************************")
			print ("You have touched the exit button")
			print ("********************************")
			self.cap.release()
			cv2.destroyAllWindows()
			exit()



		if (self.save_button == False):
			self.buttons(self.save_point, (255, 0, 0), 'Sav', (255, 255, 255))
			self.save_button = self.button_selected(self.save_point)

		else:
			self.buttons(self.save_point, (0, 255, 0), 'Sav', (255, 255, 255))

			if (self.action_done == False):
				cv2.imwrite('./save_images/' + str(self.image_num) + '.png', self.current_frame2)
				self.image_num += 1
				self.action_done = True

			if (self.button_left(self.save_point)):
				self.save_button = False
				self.action_done = False



		if (self.timer_button == False):
			self.buttons(self.timer_point, (255, 0, 0), 'Tmr', (255, 255, 255))
			self.timer_button = self.button_selected(self.timer_point)
		
		else:
			self.buttons(self.timer_point, (0, 255, 0), 'Tmr', (255, 255, 255))
			if self.timer_button_clock_flag == False:
				self.timer_button_clock_flag = True
				self.timer_button_clock_value = time.time()

			if (self.timer_button_clock_flag):
				count = np.int(time.time() - self.timer_button_clock_value)
				if (count < self.timer_value):
					self.current_frame = write_data3(self.current_frame, 'Saving image in ' + str(int(self.timer_value - count)) + ' secs ', 0.0, 0.4, 1.0, 0.15, 0.04, 0.12, 2, 2, (255,255,255))
				else:
					cv2.imwrite('./save_images/' + str(self.image_num) + '.png', self.current_frame2)
					self.image_num += 1
					self.timer_button_clock_flag = False
					self.timer_button = False



			


		self.view_frame()
		
		

		



