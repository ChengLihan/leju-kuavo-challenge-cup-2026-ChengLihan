#!/usr/bin/env python3
"""Body control utils: bend/tilt, squat, stand, step_back for Scene3 lower tray pick."""

import time

import rospy
from geometry_msgs.msg import Twist


_CMD_POSE_PUB = None


def _get_cmd_pose_pub():
    global _CMD_POSE_PUB
    if _CMD_POSE_PUB is None:
        _CMD_POSE_PUB = rospy.Publisher("/cmd_pose", Twist, queue_size=10)
    return _CMD_POSE_PUB


def stop_base(cmd_pub, count=5):
    msg = Twist()
    for _ in range(int(count)):
        cmd_pub.publish(msg)
        rospy.sleep(0.05)


def bend_forward(angle_rad=0.5, duration=2.0):
    """Tilt torso forward by angle_rad radians (positive = bend forward)."""
    import math
    angle_deg = math.degrees(float(angle_rad))
    rospy.loginfo("BodyControl: bending forward %.1fdeg over %.1fs", angle_deg, duration)
    pose_pub = _get_cmd_pose_pub()
    rate = rospy.Rate(20)
    end_time = time.time() + float(duration)
    while time.time() < end_time and not rospy.is_shutdown():
        elapsed = time.time() - (end_time - float(duration))
        frac = min(1.0, elapsed / float(duration))
        msg = Twist()
        msg.angular.y = float(angle_rad) * frac
        pose_pub.publish(msg)
        rate.sleep()
    msg = Twist()
    msg.angular.y = float(angle_rad)
    pose_pub.publish(msg)
    rospy.sleep(0.5)


def bend_and_squat(angle_rad=0.5, height_delta=-0.30, duration=2.0):
    """Bend forward WHILE squatting — angular.y + linear.z simultaneously."""
    import math
    angle_deg = math.degrees(float(angle_rad))
    rospy.loginfo("BodyControl: bending %.1fdeg + squatting %.2fm over %.1fs", angle_deg, float(height_delta), duration)
    pose_pub = _get_cmd_pose_pub()
    rate = rospy.Rate(20)
    end_time = time.time() + float(duration)
    while time.time() < end_time and not rospy.is_shutdown():
        elapsed = time.time() - (end_time - float(duration))
        frac = min(1.0, elapsed / float(duration))
        msg = Twist()
        msg.angular.y = float(angle_rad) * frac
        msg.linear.z = float(height_delta) * frac
        pose_pub.publish(msg)
        rate.sleep()
    msg = Twist()
    msg.angular.y = float(angle_rad)
    msg.linear.z = float(height_delta)
    pose_pub.publish(msg)
    rospy.sleep(0.5)


def stand_straight(duration=2.0, cmd_vel_topic="/cmd_vel"):
    """Return torso to upright (angular.y=0) and height to normal (linear.z=0)."""
    rospy.loginfo("BodyControl: standing straight over %.1fs", duration)
    cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    pose_pub = _get_cmd_pose_pub()
    rate = rospy.Rate(20)
    end_time = time.time() + float(duration)
    while time.time() < end_time and not rospy.is_shutdown():
        msg = Twist()
        msg.angular.y = 0.0
        msg.linear.z = 0.0
        pose_pub.publish(msg)
        rate.sleep()
    stop_base(cmd_pub, count=5)
    rospy.sleep(0.5)


def squat(height_delta=-0.15, duration=2.0):
    rospy.loginfo("BodyControl: squatting %.2fm over %.1fs", height_delta, duration)
    pose_pub = rospy.Publisher("/cmd_pose", Twist, queue_size=10)
    rate = rospy.Rate(20)
    end_time = time.time() + float(duration)
    while time.time() < end_time and not rospy.is_shutdown():
        elapsed = time.time() - (end_time - float(duration))
        frac = min(1.0, elapsed / float(duration))
        msg = Twist()
        msg.linear.z = float(height_delta) * frac
        pose_pub.publish(msg)
        rate.sleep()
    msg = Twist()
    msg.linear.z = float(height_delta)
    pose_pub.publish(msg)
    rospy.sleep(0.5)


def stand(duration=2.0, cmd_vel_topic="/cmd_vel"):
    rospy.loginfo("BodyControl: standing up over %.1fs", duration)
    cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    pose_pub = rospy.Publisher("/cmd_pose", Twist, queue_size=10)
    rate = rospy.Rate(20)
    end_time = time.time() + float(duration)
    while time.time() < end_time and not rospy.is_shutdown():
        msg = Twist()
        msg.linear.z = 0.0
        pose_pub.publish(msg)
        rate.sleep()
    stop_base(cmd_pub, count=5)
    rospy.sleep(0.5)


def step_back(distance=0.30, speed=0.10, timeout=10.0, cmd_vel_topic="/cmd_vel"):
    cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    speed = max(0.02, min(float(speed), 0.20))
    duration = min(float(timeout), max(0.5, float(distance) / speed))
    rospy.loginfo(
        "BodyControl: stepping back %.2fm at %.2fm/s for %.1fs",
        distance,
        speed,
        duration,
    )
    msg = Twist()
    msg.linear.x = -speed
    rate = rospy.Rate(20)
    end_time = time.time() + duration
    while time.time() < end_time and not rospy.is_shutdown():
        cmd_pub.publish(msg)
        rate.sleep()
    stop_base(cmd_pub)
    rospy.sleep(0.5)
