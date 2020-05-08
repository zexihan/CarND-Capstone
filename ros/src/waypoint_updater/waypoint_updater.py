#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 25 # init = 200 Number of waypoints we will publish. You can change this number
MAX_DECEL = 2.0 # 

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        # rospy.loginfo('Initializing WaypointUpdater node')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_lane = None
        self.stopline_wp_idx = -1
        self.pose = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.loop()

    def loop(self):
        # Rate attempts to loop at 50Hz since our final waypoints will be
        # published to waypoints_follower, autoware code that runs at 30Hz

        # set ros publishing frequency to 50Hz
        # rospy.loginfo('Setting loop rate to 50hz')
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_id()
                # rospy.loginfo('closest waypoint index = %s', closest_waypoint_idx)
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_id(self):
        # Gets closest waypoint in front of car
        # rospy.loginfo('Getting closest waypoint id')
        x = self.pose.pose.position.x
        # rospy.loginfo('position x = %s', x)
        y = self.pose.pose.position.y
        # rospy.loginfo('position y = %s', y)
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        # rospy.loginfo('Get closest and prev coord')
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        # rospy.loginfo('Hyperplane through closest_coords')
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        # rospy.loginfo('Check is closest waypoint behind or in front of car')
        if val > 0:
            # rospy.loginfo('Closest waypoint behind car, get next coord')
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_idx):
        # rospy.loginfo('Publishing 200 closest waypoints in front of car')
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_id()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0)
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            # Two waypoints back from line so front of car stops at line
            # stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # Store car's incoming position 
        # rospy.loginfo("Receiving car's position")
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Receive car's incoming waypoints, create 2d waypoints, construct 
        # KDTree for searching for which waypoint is closest to the car
        # Once the waypoint_tree is constructed, then select simulator track
        # rospy.loginfo('Constructing waypoint tree from base waypoints')
        self.base_lane = waypoints
        # rospy.loginfo('Check is 2d waypoints created, else create it for KDTree')
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # rospy.loginfo('waypoints[0].pose.pose.position.x = %s', waypoints.waypoints[0].pose.pose.position.x)
            self.waypoint_tree = KDTree(self.waypoints_2d)
            # rospy.loginfo('waypoint_tree data = %s', self.waypoint_tree.data)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data
        # rospy.loginfo("stopline_wp_idx = %s", self.stopline_wp_idx)
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
