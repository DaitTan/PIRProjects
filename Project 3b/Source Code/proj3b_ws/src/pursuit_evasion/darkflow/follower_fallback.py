#!/usr/bin/env python
import os
print(os.getcwd())
import roslib
import sys
import rospy
import cv2
from darkflow.net.build import TFNet
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply

import time
import os
import sys
import numpy as np

PI = 3.1415926535897

options = {"model": "/home/daittan/nsvr3b_ws/src/pursuit_evasion/darkflow/cfg/v1/yolo-tiny.cfg", "load": "/home/daittan/nsvr3b_ws/src/pursuit_evasion/darkflow/bin/yolo-tiny.weights", "GPU": 1.0}



PI = 3.1415926535897


def getConf(elem):
	return elem[4]

def getAngle(pix):
    angle = (640.0-pix)/640.0
    # return angle*(60.0)-(30.0)
    return (angle*(60.0*(PI/180.0)))-(30.0*(PI/180.0))



class person_follower:
    def __init__(self):
        self.image_pub = rospy.Publisher("/CV_image", Image, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size=1)
        
        self.bridge = CvBridge()
        self.net = TFNet(options)
        self.ns = rospy.get_namespace()
        self.velMsg = Twist()
        topic_1 = self.ns + 'tb3_0/move_base/result'
        topic_2 = self.ns + 'tb3_0/move_base_simple/goal'
        print(topic_1)
        # self.sub = rospy.Subscriber(topic_1, MoveBaseActionResult, self.statusCB, queue_size=10)
        self.pub = rospy.Publisher(topic_2, PoseStamped, queue_size=10)
        
        self.sub_tb = rospy.Subscriber("/tb3_0/amcl_pose", PoseWithCovarianceStamped, self.amclPoseCallback)
        image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)
        
        
        # self.goalId = 0
        # self.goalMsg = PoseStamped()
        # self.goalMsg.header.stamp = rospy.Time.now()
        # self.goalMsg.header.frame_id = "map"
        # self.goalMsg.pose.position.x = coordinate[0]
        # self.goalMsg.pose.position.y = coordinate[1]
        # self.goalMsg.pose.position.z = coordinate[2]
        # time.sleep(1)
        # self.goalMsg.pose.orientation.x = quart[0]
        # self.goalMsg.pose.orientation.y = quart[1]
        # self.goalMsg.pose.orientation.z = quart[2]
        # self.goalMsg.pose.orientation.w = quart[3]
        # self.pub.publish(self.goalMsg)
        

        
        
        # params & variables
        


    def statusCB(self, point, quart):
        self.goalId = 0
        self.goalMsg = PoseStamped()
        self.goalMsg.header.stamp = rospy.Time.now()
        self.goalMsg.header.frame_id = "map"
        self.goalMsg.pose.position.x = point[0]
        self.goalMsg.pose.position.y = point[1]
        self.goalMsg.pose.position.z = point[2]
        self.goalMsg.pose.orientation.x = quart[0]
        self.goalMsg.pose.orientation.y = quart[1]
        self.goalMsg.pose.orientation.z = quart[2]
        self.goalMsg.pose.orientation.w = quart[3]
        self.pub.publish(self.goalMsg)
        
        
            
    def amclPoseCallback(self,pose):
        self.pursuerCurrentLoc = [pose.pose.pose.position.x, pose.pose.pose.position.y]
        self.pursuerOrientation = [pose.pose.pose.orientation.x,pose.pose.pose.orientation.y,pose.pose.pose.orientation.z,pose.pose.pose.orientation.w]

    def callback(self, ros_image):
		#print("!callback initiated count = " + str(self.count))
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        

        results = self.net.return_predict(cv_image)
        modResults, cv_image = self.checkForPerson(results, cv_image)
		
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image,"bgr8"))
        except CvBridgeError as e:
            print(e)
		
        		
# 		 >>> Follow the person >>>
        if modResults:
            if modResults[0]:
                self.follow_person(modResults)
            else:
                self.velMsg.linear.x = 0
                self.velMsg.angular.z = 0
        self.velMsg.linear.x = 0
        self.velMsg.angular.z = 0

		

    def getPos(self):
        rospy.wait_for_message("/tb3_0/amcl_pose", PoseWithCovarianceStamped)
        return self.pursuerCurrentLoc, self.pursuerOrientation

    def checkForPerson(self, results, input_im):
        modRes = []
        for listInd, val in enumerate(results):
            if val['label'] == "person":
                if val['confidence'] >= 0.07:
                    a = [val['topleft']['x'], val['topleft']['y'],val['bottomright']['x'], val['bottomright']['y'], val['confidence']]
                    print(val['label'])
                    modRes.append(a)
        if not modRes:
            outIm = input_im
        else:
            if not modRes[0]:
                outIm = input_im
            else:
                modRes.sort(key=getConf, reverse=True)
                topResult = modRes[0]
				
                topLeft = (topResult[0],topResult[1])
				
                bottomRight = (topResult[2],topResult[3])

                outIm = cv2.rectangle(input_im,topLeft, bottomRight, (255,0,0), 2)
			
        return modRes, outIm
	
    
        
    def forward(self, distance, speed):
    	# This function moves the turtleBot forward for given distance with some speed.
    	# Apparently, it just moves forward. No other direction
    
    	self.velMsg.linear.x = abs(speed)
    	self.velMsg.linear.y = 0
    	self.velMsg.linear.z = 0
    	self.velMsg.angular.x = 0
    	self.velMsg.angular.y = 0
    	self.velMsg.angular.z = 0
    
    	t0 = rospy.Time.now().to_sec()
    	current_distance = 0
    
    	while(current_distance <= distance):
            self.cmd_vel_pub.publish(self.velMsg)
            t1=rospy.Time.now().to_sec()
            current_distance= speed*(t1-t0)
            
        self.velMsg.linear.x = 0
        self.cmd_vel_pub.publish(self.velMsg)
    
    def rotate(self, angle, speed):
    	# The turtle rotates for angle with a given angular speed.
    	# If speed is positive, movement is anti-clockwise
    	# If speed is negative, movement is clockwise
    	
    	angular_speed = speed * 2*PI/360
        angle = angle * 2*PI/360
    	
    
    	self.velMsg.linear.x = 0
    	self.velMsg.linear.y = 0
    	self.velMsg.linear.z = 0
    	self.velMsg.angular.x = 0
    	self.velMsg.angular.y = 0
        self.velMsg.angular.z = -1*(angular_speed)
    	
    
    	t0 = rospy.Time.now().to_sec()
    	current_angle = 0
    	
    
    	while(abs(current_angle) <= abs(angle)):
    		self.cmd_vel_pub.publish(self.velMsg)
    		t1=rospy.Time.now().to_sec()
    		current_angle= angular_speed*(t1-t0)
    		
    	self.velMsg.angular.z = 0
        # self.cmd_vel_pub.publish(self.velMsg)

    def follow_person(self, modResults):

        box_center = (modResults[0][0] + modResults[0][2])/2.0
        
        angle = getAngle(box_center)
        # a,b = self.getPos()
        # print(a)
        # print(angle)
        # x = a[0] + np.sin(angle)
        # y = a[1] + np.cos(angle)
        self.rotate(angle,1)
        self.forward(1,1)
        self.velMsg.linear.x = 0
        self.velMsg.angular.z = 0
        self.cmd_vel_pub.publish(self.velMsg)
        # r_quat = quaternion_from_euler(0,0,angle)
		
        # newQuat = quaternion_multiply(b,r_quat)

        # coordinate = (x,y,0.0)
        # print("************Moving to****************")
        # print(coordinate)
        # print(newQuat)
        
        # self.statusCB(coordinate,newQuat)
        # print("************Moving end****************")
		

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
