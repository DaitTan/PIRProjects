
from __future__ import print_function


import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from darkflow.net.build import TFNet
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError


options = {"model": "/home/daittan/Documents/proj3b_ws/src/pursuit_evasion/darkflow/cfg/yolov2.cfg", "load": "/home/daittan/Documents/proj3b_ws/src/pursuit_evasion/darkflow/bin/yolov2.weights"}


import os
import sys
import numpy as np
import tensorflow as tf




class person_follower:

	def __init__(self):
		print('ROS_OpenCV_bridge initialized')
		self.image_pub = rospy.Publisher("/CV_image", Image, queue_size=5)
		self.cmd_vel_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size=5)
		self.bridge = CvBridge()
		self.net = TFNet(options)
		image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)

		print("Subscribed to image_raw")


	'''
		This methdod performs these steps:
		1. Converts a ROS Image to a CV image
		2. runs an object detection inference on the image
		3. Draws the inference on the image
		4. publishes the drawn inference
		5. Follows the human
	'''
	def callback(self, ros_image):
		print('!callback initiated')
		try:
			cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError as e:
			print(e)
		# print(type(cv_image))     # <type 'numpy.ndarray'>

		results = self.net.return_predict(cv_image)
		modResults = self.checkForPerson(results)
		self.draw_output(cv_image, modResults)
		

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

		# >>> Follow the person >>>
		#self.follow_person(output_dict)

	def checkForPerson(self, results):
		modRes = []
		for listInd, val in enumerate(result):
			if val['label'] == "person":
				if val['confidence'] >= 0.5:
					a = [val['topleft']['x'], val['topleft']['y'],val['bottomright']['x'], val['bottomright']['y'], val['confidence'], val['label']]
					modRes.append(a)
		return modRes

	def run_inference_for_single_image(self, image):
		image = np.asarray(image)
		# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
		input_tensor = tf.convert_to_tensor(image)
		# The model expects a batch of images, so add an axis with `tf.newaxis`.
		input_tensor = input_tensor[tf.newaxis, ...]

		# Run inference
		output_dict = self.detection_model(input_tensor)

		# All outputs are batches tensors.
		# Convert to numpy arrays, and take index [0] to remove the batch dimension.
		# We're only interested in the first num_detections.
		num_detections = int(output_dict.pop('num_detections'))
		output_dict = {key: value[0, :num_detections].numpy()
						for key, value in output_dict.items()}
		output_dict['num_detections'] = num_detections

		# detection_classes should be ints.
		output_dict['detection_classes'] = output_dict['detection_classes'].astype(
			np.int64)

		# Handle models with masks:
		if 'detection_masks' in output_dict:
			# Reframe the the bbox mask to the image size.
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					output_dict['detection_masks'], output_dict['detection_boxes'],
					image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
											tf.uint8)
			output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

		return output_dict


	def draw_output(self, np_img, results):
		modImg = cv2.rectangle(np_img, (results[0],results[1]), (results[2],results[3]), (255,0,0), 2)
		return modImg

	
	def follow_person(self, output_dict):
		'''
			only publish when we find a person
		'''
		move_cmd = Twist()		# all values are 0.0 by default

		index = -1
		for i in range(len(output_dict['detection_classes'])):
			if (output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.4):		# if we detected a person
				index = i	# keep the index
				break		# break the for loop

		if (index > -1):   # if we found a person
			box = output_dict['detection_boxes'][index]
			box_center = (box[1] + box[3])/2.0

			# 0.872 radians is 50 degrees
			move_cmd.angular.z = -1.1 * (box_center - 0.5)	# subtract 0.5 so that it is a value between -0.5 and 0.5
			move_cmd.linear.x = 0.21

			debug = False
			debug = True
			if(debug):
				print('box_center: ', box_center)
				print('angular.z: {:.2f}'.format(move_cmd.angular.z))
					
			# >>> publish the movement speeds >>>
			self.cmd_vel_pub.publish(move_cmd)

				

def main(args):
	rospy.init_node('person_follower', anonymous=False)  # we only need one of these nodes so make anonymous=False
	pf = person_follower()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

