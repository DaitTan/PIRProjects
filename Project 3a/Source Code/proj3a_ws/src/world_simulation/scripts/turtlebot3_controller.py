#!/usr/bin/env python

# this code is adapted from https://github.com/ROBOTIS-GIT/turtlebot3/blob/master/turtlebot3_example/nodes/turtlebot3_pointop_key
# I was able to find the source code thanks to this thread https://answers.ros.org/question/154876/no-source-code-cpp-file-for-turtlebot/

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
import numpy as np



class turtlebot3_controller():
    def __init__(self):
        #rospy.init_node('turtlebot3_controller', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        self.rate = rospy.Rate(10)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
    
    
    def goToPoint(self,goal_x,goal_y,goal_z=None, linear_speed=1):
        (position, rotation) = self.get_odom()
        last_rotation = 0
        angular_speed = 1
        if goal_z > 180 or goal_z < -180:
            print("incorrect z range.")
            return
        goal_z = np.deg2rad(goal_z)
        goal_distance = sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
        distance = goal_distance

        position = Point()
        move_cmd = Twist()      # geometry_msgs/Twist
        
        # move to the new (goal_x, goal_y) location
        while distance > 0.05:
            (position, rotation) = self.get_odom()
            x_start = position.x
            y_start = position.y
            path_angle = atan2(goal_y - y_start, goal_x- x_start)

            if path_angle < -pi/4 or path_angle > pi/4:
                if goal_y < 0 and y_start < goal_y:
                    path_angle = -2*pi + path_angle
                elif goal_y >= 0 and y_start > goal_y:
                    path_angle = 2*pi + path_angle
            if last_rotation > pi-0.1 and rotation <= 0:
                rotation = 2*pi + rotation
            elif last_rotation < -pi+0.1 and rotation > 0:
                rotation = -2*pi + rotation
            move_cmd.angular.z = angular_speed * path_angle-rotation

            distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))

            # move faster if we are farther away from the goal
            if(distance < 0.3):
                move_cmd.linear.x = 0.2
            else:
                move_cmd.linear.x = 0.4

            if move_cmd.angular.z > 0:
                move_cmd.angular.z = min(move_cmd.angular.z, 1.5)   
            else:
                move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

            last_rotation = rotation
            self.cmd_vel.publish(move_cmd)
            self.rate.sleep()
        (position, rotation) = self.get_odom()

        # rotate to the "goal_z" degrees IF requested
        if(True):
            while abs(rotation - goal_z) > 0.05:
                (position, rotation) = self.get_odom()
                if goal_z >= 0:
                    if rotation <= goal_z and rotation >= goal_z - pi:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = 0.5
                    else:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = -0.5
                else:
                    if rotation <= goal_z + pi and rotation > goal_z:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = -0.5
                    else:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = 0.5
                self.cmd_vel.publish(move_cmd)
                self.rate.sleep()


        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        self.rate.sleep()


    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

