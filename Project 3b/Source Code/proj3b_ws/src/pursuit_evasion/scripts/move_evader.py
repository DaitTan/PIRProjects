#!/usr/bin/env python2.7
# Copyright 2017 HyphaROS Workshop.
# Developer: HaoChih, LIN (hypha.ros@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rospy
import string
import math
import time
import sys

from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseActionResult
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped


class MultiGoals:
    def __init__(self, goalListX, goalListY, retry, map_frame):
        self.ns = rospy.get_namespace()
        topic_1 = self.ns + 'move_base/result'
        topic_2 = self.ns + 'move_base_simple/goal'
        self.sub = rospy.Subscriber(topic_1, MoveBaseActionResult, self.statusCB, queue_size=10)
        self.pub = rospy.Publisher(topic_2, PoseStamped, queue_size=10)
        # params & variables
        self.goalListX = goalListX
        self.goalListY = goalListY
        self.retry = retry
        self.goalId = 0
        self.goalMsg = PoseStamped()
        self.goalMsg.header.frame_id = map_frame
        self.goalMsg.pose.orientation.z = 0.0
        self.goalMsg.pose.orientation.w = 1.0
        # Publish the first goal
        time.sleep(1)
        self.goalMsg.header.stamp = rospy.Time.now()
        self.goalMsg.pose.position.x = self.goalListX[self.goalId]
        self.goalMsg.pose.position.y = self.goalListY[self.goalId]
        self.pub.publish(self.goalMsg)
        rospy.loginfo("Initial goal published! Goal ID is: %d", self.goalId)
        self.goalId = self.goalId + 1

    def statusCB(self, data):
        if data.status.status == 3:  # reached
            self.goalMsg.header.stamp = rospy.Time.now()
            self.goalMsg.pose.position.x = self.goalListX[self.goalId]
            self.goalMsg.pose.position.y = self.goalListY[self.goalId]
            self.pub.publish(self.goalMsg)
            rospy.loginfo("Reached: Goal ID is: %d", self.goalId)
            if self.goalId < (len(self.goalListX) - 1):
                self.goalId = self.goalId + 1
            else:
                self.goalId = 0


if __name__ == "__main__":
    try:
        # ROS Init    
        rospy.init_node('multi_goals', anonymous=True)

        # Get params
        goalListX = rospy.get_param('~goalListX', '[2.0, 2.0]')
        goalListY = rospy.get_param('~goalListY', '[2.0, 4.0]')
        map_frame = "map"  # rospy.get_param('~map_frame', 'map' )
        retry = 1  # rospy.get_param('~retry', '1')

        goalListX = goalListX.replace("[", "").replace("]", "")
        goalListY = goalListY.replace("[", "").replace("]", "")
        goalListX = [float(x) for x in goalListX.split(",")]
        goalListY = [float(y) for y in goalListY.split(",")]
        print(goalListX)
        print(goalListY)
        # goalListX = [2.69, 4.68, 3.6, 1.13, -0.83, -2.47, -5.49, -6.28, -4.24]
        # goalListY = [1.34, -1.5, -3.76, -4.48, -2.86, 0.76, 1.85, -1.19, -3.73]

        if len(goalListX) == len(goalListY) & len(goalListY) >= 2:
            # Constract MultiGoals Obj
            rospy.loginfo("Multi Goals Executing...")
            mg = MultiGoals(goalListX, goalListY, retry, map_frame)
            rospy.spin()
        else:
            rospy.errinfo("Lengths of goal lists are not the same")
    except KeyboardInterrupt:
    	print("shutting down")
