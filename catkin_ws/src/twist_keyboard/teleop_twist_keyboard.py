#!/usr/bin/env python3
# This file is adopted from
# https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py

"""
How to use this script
- <Esc> for exit
- Arrow keys for moving. 
- w,s for increasing and decreasing linear velocity
- a,d for increasing and decreasing angular velocity
- You can move the robot while changing velocities.
- Once letting go of the arrow keys, the robot will stop.

How this script works:
1. Keyboard event comes in -> if it's char or special. add to CHARS, SPECIALS.
    - because there will be a release, we don't worry about SPECIALS overflowing.
    - So just parse through all keys
2. Then, for all chars and specials:
    - grab their correspnding action, from dict
    - char: determines the absolute value of a vel
        s: vel = abs(min([0,0], vel-linear_increment))
        w: vel = max(MAX_VEL, vel+linear_increment)
    - arrow: determine which vel is selected, and sign
        return_vel = np.zeros(2)
        up_arrow: vel[0] + return_vel[0]

Caveat: 
    - The process will read your keystrokes even when you're not on the main console.

Inputs:
    - Keyboard events 
Outputs:
    - Publisher /ROS_TOPIC/CMD_VEL

How to run this script, in a remote container
    1. on hostmachine, xhost local:root
    2. SSH -Y ...
    3. sudo docker run --name my_ros_container --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $HOME/.Xauthority:/root/.Xauthority -e XAUTHORITY=/root/.Xauthority -v /home/ricojia/software/The-Dream-Robot/:/home/The-Dream-Robot -v ~/.ssh:/root/.ssh   -it   --network="host"   --privileged   ricojia/rpi-dream-mobile-platform
    4. sudo_ros_preserve_env rosrun dream_mobile_platform keyboard_teleop.py
    5. Hit <ESC to quit>
"""

from pynput import keyboard
import numpy as np
import typing
import sys

import rospy
from geometry_msgs.msg import Twist


# Convention: [abs(lin_vel), abs(ang_vel)]
MIN_VEL = np.zeros(2)
MAX_VEL = np.array([1.0, 3.14])
LIN_INCREMENT = np.array([0.02, 0.0])
ANG_INCREMENT = np.array([0.0, 0.1])
CHARS = set()
SPECIALS = set()

vel = np.zeros(2)

# Note on why this dict takes in a str:
# The pynput library currently considers two Keycode objects of the same char DIFFERENT.
# That's not an expected behavior
CHAR_ACTIONS: typing.Dict[str, typing.Callable[[np.ndarray], np.ndarray]] = {
    "w": lambda vel: np.minimum(vel + LIN_INCREMENT, MAX_VEL),
    "s": lambda vel: np.maximum(vel - LIN_INCREMENT, MIN_VEL),
    "a": lambda vel: np.minimum(vel + ANG_INCREMENT, MAX_VEL),
    "d": lambda vel: np.maximum(vel - ANG_INCREMENT, MIN_VEL),
}

SPECIALS_ACTIONS: typing.Dict[
    keyboard.Key, typing.Callable[[np.ndarray, np.ndarray], np.ndarray]
] = {
    # these actions uses "masks" on the velocity to control.
    keyboard.Key.up: lambda return_vel, vel: return_vel + np.array([1.0, 0]) * vel,
    keyboard.Key.down: lambda return_vel, vel: return_vel + np.array([-1.0, 0]) * vel,
    keyboard.Key.left: lambda return_vel, vel: return_vel + np.array([0.0, 1.0]) * vel,
    keyboard.Key.right: lambda return_vel, vel: return_vel
    + np.array([0.0, -1.0]) * vel,
}


def clear_line():
    """Clear the current line on stdout so the pressed char won't show"""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def add_or_remove_key(
    key: typing.Union[keyboard.Key, keyboard.KeyCode], is_key_pressed: bool
):
    """based on the key type, and if it's pressed, we will add it and remove it to the set"""
    global CHARS
    global SPECIALS
    if isinstance(key, keyboard.Key):
        target_set = SPECIALS
        new_member = key
    elif isinstance(event.key, keyboard.KeyCode):
        target_set = CHARS
        new_member = key.char
    else:
        return
    if is_key_pressed:
        target_set.add(new_member)
    else:
        target_set.discard(new_member)


# Calculate velocity using abs values. Their signs are determined by the arrow keys
def keyboard_event_analyzer(event, commanded_wheel_vel_pub):
    clear_line()
    add_or_remove_key(event.key, isinstance(event, keyboard.Events.Press))
    global vel
    # adding all actions together
    for action, action_func in CHAR_ACTIONS.items():
        if action in CHARS:
            vel = action_func(vel)
    return_vel = np.zeros(2)
    for action, action_func in SPECIALS_ACTIONS.items():
        if action in SPECIALS:
            return_vel = action_func(return_vel, vel)

    msg = Twist()
    if CHARS:
        pass
    if SPECIALS:
        pass
        msg.linear.x = return_vel[0]
        msg.angular.z = return_vel[1]
        commanded_wheel_vel_pub.publish(msg)
    elif not SPECIALS and not CHARS:
        msg.linear.x = return_vel[0]
        msg.angular.z = return_vel[1]
        commanded_wheel_vel_pub.publish(msg)


if __name__ == "__main__":
    # The event listener will be running in this block
    node_name = "keyboard_teleop"
    rospy.init_node(node_name)
    commanded_wheel_vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=5
    )
    with keyboard.Events() as events:
        for event in events:
            if event.key == keyboard.Key.esc:
                break
            else:
                keyboard_event_analyzer(event, commanded_wheel_vel_pub)