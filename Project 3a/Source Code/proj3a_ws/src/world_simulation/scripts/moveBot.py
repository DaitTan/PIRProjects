#!/usr/bin/env python
# license removed for brevity

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist

PI = 3.1415926535897

def forward(distance, speed, vel_msg, velocity_publisher):
	# This function moves the turtleBot forward for given distance with some speed.
	# Apparently, it just moves forward. No other direction

	vel_msg.linear.x = abs(speed)
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0
	vel_msg.angular.z = 0

	t0 = rospy.Time.now().to_sec()
	current_distance = 0

	while(current_distance <= distance):
		velocity_publisher.publish(vel_msg)
	    	t1=rospy.Time.now().to_sec()
	    	current_distance= speed*(t1-t0)
	    	
			
		
	vel_msg.linear.x = 0

def rotate(angle, speed, vel_msg, velocity_publisher):
	# The turtle rotates for angle with a given angular speed.
	# If speed is positive, movement is anti-clockwise
	# If speed is negative, movement is clockwise
	
	angular_speed = speed * 2*PI/360
	angle = angle * 2*PI/360
	

	vel_msg.linear.x = 0
	vel_msg.linear.y = 0
	vel_msg.linear.z = 0
	vel_msg.angular.x = 0
	vel_msg.angular.y = 0
	vel_msg.angular.z = angular_speed
	

	t0 = rospy.Time.now().to_sec()
	current_angle = 0
	

	while(abs(current_angle) <= abs(angle)):
		velocity_publisher.publish(vel_msg)
	    	t1=rospy.Time.now().to_sec()
	    	current_angle= angular_speed*(t1-t0)
		
	vel_msg.angular.z = 0

def drawK(velocity_publisher):
	cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
	vel_msg = Twist()
	angularSpeed = 20
	linearSpeed = 0.1

	forward(0.7, linearSpeed, vel_msg, velocity_publisher)
	rotate(180, angularSpeed, vel_msg, velocity_publisher)
	forward(0.35, linearSpeed, vel_msg, velocity_publisher)
	rotate(45, angularSpeed, vel_msg, velocity_publisher)
	forward(0.35, linearSpeed, vel_msg, velocity_publisher)
	rotate(180, angularSpeed, vel_msg, velocity_publisher)
	forward(0.35, linearSpeed, vel_msg, velocity_publisher)
	rotate(90, -1*angularSpeed, vel_msg, velocity_publisher)
	forward(0.35, linearSpeed, vel_msg, velocity_publisher)
	rotate(180, angularSpeed, vel_msg, velocity_publisher)
	forward(0.35, linearSpeed, vel_msg, velocity_publisher)

def drawBigK(velocity_publisher):
	cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
	vel_msg = Twist()
	angularSpeed = 20
	linearSpeed = 0.1

	forward(1.0, linearSpeed, vel_msg, velocity_publisher)
	rotate(180, angularSpeed, vel_msg, velocity_publisher)
	forward(0.5, linearSpeed, vel_msg, velocity_publisher)
	rotate(45, angularSpeed, vel_msg, velocity_publisher)
	forward(0.5, linearSpeed, vel_msg, velocity_publisher)
	rotate(180, angularSpeed, vel_msg, velocity_publisher)
	forward(0.5, linearSpeed, vel_msg, velocity_publisher)
	rotate(90, -1*angularSpeed, vel_msg, velocity_publisher)
	forward(0.5, linearSpeed, vel_msg, velocity_publisher)
	rotate(180, angularSpeed, vel_msg, velocity_publisher)
	forward(0.5, linearSpeed, vel_msg, velocity_publisher)


def movebase_client(coordinateList):

    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(Point(coordinateList[0], coordinateList[1], 0.000), Quaternion(0, 0,coordinateList[2],coordinateList[3]))

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
	
        return client.get_result(), coordinateList

if __name__ == '__main__':
    try:
        rospy.init_node('movebase_client_py')
	cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
	roomsDict = {"bed": [1.270,1.563,0,1],
		     "mbed" : [3.27, -0.85, -0.73,0.687],
		     "toil":[-1.967, 2.918, 1,0],
		     "mtoil":[0.99, -2.40,0,1],
 		     "gym":[-0.037,-1.176,-0.7669,0.6416],
		     "hall":[-3.09,-0.292,1,0],
		     "kithcen":[-1.663, 1.873, 0,1],
 		     "balcony":[-4.95, 1.92,0.77,0.63],
		     "origin":[0.0,0.0,0.0,1]}

	       
	result = movebase_client(roomsDict["mbed"])
	drawBigK(cmd_vel)

	result = movebase_client(roomsDict["mtoil"])
	drawK(cmd_vel)

        result = movebase_client(roomsDict["bed"])
	drawBigK(cmd_vel)

        result = movebase_client(roomsDict["kithcen"])
	drawK(cmd_vel)

        result = movebase_client(roomsDict["toil"])
	drawK(cmd_vel)

	result = movebase_client(roomsDict["gym"])
	drawK(cmd_vel)

        result = movebase_client(roomsDict["hall"])
	drawBigK(cmd_vel)

	result = movebase_client(roomsDict["balcony"])
	drawK(cmd_vel)

	result = movebase_client(roomsDict["balcony"])

        if result:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")

