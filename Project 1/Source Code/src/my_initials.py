#!/usr/bin/env python
# Parts of the function def forward() and def rotate() have been taken
# from tutorials on ROS website. 

# def forward() from http://wiki.ros.org/turtlesim/Tutorials/Moving%20in%20a%20Straight%20Line
# def rotate() from http://wiki.ros.org/turtlesim/Tutorials/Rotating%20Left%20and%20Right

# The letter K was drawn first drawn on paper to calculate al the angles and length and
# then implemented here. There are some minor angle errors since I did not have access to 
# proper tools. 


import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

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

def move():
	rospy.init_node('robot_cleaner', anonymous=True)
	velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
	vel_msg = Twist()
	angularSpeed = 45
	linearSpeed = 5

	rotate(90, -1 * 10, vel_msg, velocity_publisher)
	forward(4, linearSpeed, vel_msg, velocity_publisher)

	rotate(90, -1 * angularSpeed, vel_msg, velocity_publisher)
	forward(1, linearSpeed, vel_msg, velocity_publisher)

	rotate(90, -1*angularSpeed, vel_msg, velocity_publisher)
	forward(8, linearSpeed, vel_msg, velocity_publisher)

	rotate(90, -1 * angularSpeed, vel_msg, velocity_publisher)
	forward(1, linearSpeed, vel_msg, velocity_publisher)

	rotate(90, -1*angularSpeed, vel_msg, velocity_publisher)
	forward(3, linearSpeed, vel_msg, velocity_publisher)

	rotate(150, angularSpeed, vel_msg, velocity_publisher)
	forward(3.45, linearSpeed, vel_msg, velocity_publisher)
	
	rotate(60, -1 * angularSpeed, vel_msg, velocity_publisher)
	forward(1, linearSpeed, vel_msg, velocity_publisher)

	rotate(120, -1 * angularSpeed, vel_msg, velocity_publisher)
	forward(3, linearSpeed, vel_msg, velocity_publisher)

	rotate(45.66, angularSpeed, vel_msg, velocity_publisher)
	forward(5.557, linearSpeed, vel_msg, velocity_publisher)

	rotate(105.66, -1 * angularSpeed, vel_msg, velocity_publisher)
	forward(1, linearSpeed, vel_msg, velocity_publisher)
	
	rotate(74.33, -1 * angularSpeed, vel_msg, velocity_publisher)
	forward(4.34, linearSpeed, vel_msg, velocity_publisher)

	rotate(134.33, angularSpeed, vel_msg, velocity_publisher)
	forward(1.118, linearSpeed, vel_msg, velocity_publisher)
	
	rotate(150, -1*angularSpeed, vel_msg, velocity_publisher)
	forward(0.5, linearSpeed, vel_msg, velocity_publisher)
	
	rospy.is_shutdown()
	#Force the robot to stop
	velocity_publisher.publish(vel_msg)

if __name__ == '__main__':
    try:
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass
