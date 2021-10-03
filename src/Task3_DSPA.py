#!/usr/bin/env python
import rospy
import numpy as np

from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse
import json

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


def dump_action_table(action_table, filename):
    """dump the MDP policy into a json file

    Arguments:
        action_table {dict} -- your mdp action table. It should be of form {'1,2,0': (1, 0), ...}
        filename {str} -- output filename
    """
    tab = dict()
    for k, v in action_table.items():
        key = [str(i) for i in k]
        key = ','.join(key)
        tab[key] = v

    with open(filename, 'w') as fout:
        json.dump(tab, fout)


class Planner:
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        """init function of the base planner. You should develop your own planner
        using this class as a base.

        For standard mazes, width = 200, height = 200, resolution = 0.05.
        For COM1 map, width = 2500, height = 983, resolution = 0.02

        Arguments:
            world_width {int} -- width of map in terms of pixels
            world_height {int} -- height of map in terms of pixels
            world_resolution {float} -- resolution of map

        Keyword Arguments:
            inflation_ratio {int} -- [description] (default: {3})
        """
        rospy.init_node('planner')
        self.map = None
        self.pose = None
        self.goal = None
        self.path = None
        self.action_seq = None  # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution
        self.inflation_ratio = inflation_ratio
        self.map_callback()
        self.sb_obs = rospy.Subscriber('/scan', LaserScan, self._obs_callback)
        self.sb_pose = rospy.Subscriber(
            '/base_pose_ground_truth', Odometry, self._pose_callback)
        self.sb_goal = rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self._goal_callback)
        self.controller = rospy.Publisher(
            '/mobile_base/commands/velocity', Twist, queue_size=10)
        rospy.sleep(1)

    def map_callback(self):
        """Get the occupancy grid and inflate the obstacle by some pixels. You should implement the obstacle inflation yourself to handle uncertainty.
        """
        self.map = rospy.wait_for_message('/map', OccupancyGrid).data

        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask

        map_img = np.array(self.map).reshape(self.world_height, self.world_width)
        map_max = max(self.map)
        map_min = min(self.map)
        map_bin = copy.deepcopy(map_img)
        map_bin[map_bin > 0] = 1
        map_bin[map_bin <= 0] = 0

        import cv2
        k_size = self.inflation_ratio + int(np.round(ROBOT_SIZE/self.resolution/2.*np.sqrt(2)))
        kernel = np.ones((3, 3), np.uint8)
        map_dilate = cv2.dilate(map_bin.astype(np.uint8), kernel, iterations = k_size).astype(np.int64)
        # kernel = np.ones((2*k_size+1, 2*k_size+1), np.uint8)
        # map_dilate = cv2.dilate(map_bin.astype(np.uint8), kernel).astype(np.int64)
        map_dilate[map_dilate > 0] = map_max
        map_dilate[map_dilate <= 0] = map_min
        self.map = copy.deepcopy(map_dilate)


        # you should inflate the map to get self.aug_map
        self.aug_map = copy.deepcopy(self.map)

    def _pose_callback(self, msg):
        """get the raw pose of the robot from ROS

        Arguments:
            msg {Odometry} -- pose of the robot from ROS
        """
        self.pose = msg

    def _goal_callback(self, msg):
        self.goal = msg
        self.generate_plan()

    def _get_goal_position(self):
        goal_position = self.goal.pose.position
        return (goal_position.x, goal_position.y)

    def set_goal(self, x, y, theta=0):
        """set the goal of the planner

        Arguments:
            x {int} -- x of the goal
            y {int} -- y of the goal

        Keyword Arguments:
            theta {int} -- orientation of the goal; we don't consider it in our planner (default: {0})
        """
        a = PoseStamped()
        a.pose.position.x = x
        a.pose.position.y = y
        a.pose.orientation.z = theta
        self.goal = a

    def _obs_callback(self, msg):
        """get the observation from ROS; currently not used in our planner; researve for the next assignment

        Arguments:
            msg {LaserScan} -- LaserScan ROS msg for observations
        """
        self.last_obs = msg

    def _d_from_goal(self, pose):
        """compute the distance from current pose to the goal; only for goal checking

        Arguments:
            pose {list} -- robot pose

        Returns:
            float -- distance to the goal
        """
        goal = self._get_goal_position()
        return sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)

    def _check_goal(self, pose):
        """Simple goal checking criteria, which only requires the current position is less than 0.25 from the goal position. The orientation is ignored

        Arguments:
            pose {list} -- robot post

        Returns:
            bool -- goal or not
        """
        if self._d_from_goal(pose) < 0.25:
            return True
        else:
            return False

    def create_control_msg(self, x, y, z, ax, ay, az):
        """a wrapper to generate control message for the robot.

        Arguments:
            x {float} -- vx
            y {float} -- vy
            z {float} -- vz
            ax {float} -- angular vx
            ay {float} -- angular vy
            az {float} -- angular vz

        Returns:
            Twist -- control message
        """
        message = Twist()
        message.linear.x = x
        message.linear.y = y
        message.linear.z = z
        message.angular.x = ax
        message.angular.y = ay
        message.angular.z = az
        return message

    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """
        self.action_seq = []


        def value_iteration(grid, gamma=0.9):
            # policy = [[(1, 0) for k in range(4) for i in range(grid.shape[1])] for j in range(grid.shape[0])]  # init all forward
            policy = {}
            actions = [(1, 0), (0, 1), (0, -1)]     # forward, left, right
            state_dim = 5    # forward, forward_left, forward_right, left, right
            state_pos = [[(0,1,0), (1,1,1), (-1,1,3), (0,0,1), (0,0,3)], \
                        [(1,0,0), (1,-1,1), (1,1,-1), (0,0,1), (0,0,-1)], \
                        [(0,-1,0), (-1,-1,1), (1,-1,-1), (0,0,1), (0,0,-1)], \
                        [(-1,0,0), (-1,1,-3), (-1,-1,-1), (0,0,-3), (0,0,-1)]]      #  4 directions, 5 states
            trns = [0.9, 0.05, 0.05]
            value_fn = np.zeros((grid.shape[0], grid.shape[1], 4))  # 4 directions
            action_pos = [[(0,1,0), (0,0,1), (0,0,3)], \
                        [(1,0,0), (0,0,1), (0,0,-1)], \
                        [(0,-1,0), (0,0,1), (0,0,-1)], \
                        [(-1,0,0), (0,0,-3), (0,0,-1)]]      #  4 directions, 3 actions

            epsilon = 0.001
            max_iter = 100000
            iter_flag = True
            iterations = 0
            # import pdb; pdb.set_trace()

            while iter_flag and iterations <= max_iter:
                # if iterations % 10 == 0:
                    # print("iterations", iterations)
                value_fn_prev = np.copy(value_fn)
                iter_flag = False
                iterations += 1
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        # if i==1 and j == 1:
                            # import pdb; pdb.set_trace()
                            # print(value_fn_prev[0:3,:3,0])
                        for k in range(4):
                            if not np.isinf(grid[i][j]):
                                q0 = 0.
                                # a = (1, 0) #  forward
                                nbs = [(i + state_pos[k][0][0],j + state_pos[k][0][1],k + state_pos[k][0][2]), \
                                (i + state_pos[k][1][0],j + state_pos[k][1][1],k + state_pos[k][1][2]), \
                                (i + state_pos[k][2][0],j + state_pos[k][2][1],k + state_pos[k][2][2])] # f, # f_l, # f_r

                                for ii in range(3):
                                    # if (nbs[ii][1], nbs[ii][0], j, i)==(13, 19, 12, 18):
                                    #     import pdb; pdb.set_trace()
                                    if not pass_wall(nbs[ii][1], nbs[ii][0], j, i):
                                        # print(nbs[ii][1], nbs[ii][0], j, i)
                                        q0 += trns[ii] * (reward[nbs[ii][0]][nbs[ii][1]] + gamma * value_fn_prev[nbs[ii]])
                                    else:
                                        q0 += trns[ii] * (reward[i,j] + gamma * value_fn_prev[i,j,k])
                                        # q0 += trns[ii] * (-2. + gamma * value_fn_prev[i,j,k])
                                q = [q0]
                                # a = (0, 1) #  left
                                nb = (i,j,0) if k == 3 else (i,j,k+1)
                                q.append(reward[nb[0], nb[1]] + gamma * value_fn_prev[nb])  # should reward be 0?
                                # a = (0, -1) #  right
                                nb = (i,j,3) if k == 0 else (i,j,k-1)
                                q.append(reward[nb[0], nb[1]] + gamma * value_fn_prev[nb])  # should reward be 0?

                                v = max(q)
                                if abs(v - value_fn_prev[i][j][k]) >= epsilon:
                                    iter_flag = True
                                value_fn[i][j][k] = v
            # print("value_fn")
            # print(value_fn[:,:,0])
            # print(value_fn[:,:,1])
            # print(value_fn[:,:,2])
            # print(value_fn[:,:,3])
            # print(value_fn[:3,1:1+3,0])
            # print(value_fn[:3,1:1+3,1])
            # print(value_fn[:3,1:1+3,2])
            # print(value_fn[:3,1:1+3,3])
            # print(value_fn[0,2,3])

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    # if i==1 and j==3:
                    #     import pdb; pdb.set_trace()
                    if not np.isinf(grid[i][j]):
                        for k in range(4):
                            action_values = {}
                            for ii in range(3):
                                a = actions[ii]
                                nb = (i+action_pos[k][ii][0],j+action_pos[k][ii][1],k+action_pos[k][ii][2])
                                if not pass_wall(nb[1], nb[0], j, i):
                                    action_values[a] = value_fn[nb]
                            if not action_values:
                                import pdb; pdb.set_trace()
                            policy[(j,i,k)] = max(action_values, key=action_values.get)
                    else:       # if accidently bump the wall
                        nb_max = -np.inf
                        nb_dir = 0
                        for ii, kk in enumerate([(0,1), (1,0), (0,-1), (-1,0)]):
                            if (j+kk[1])/self.resolution >= self.world_width or (j+kk[1]) < 0 or (i+kk[0])/self.resolution >= self.world_height or (i+kk[0]) < 0:
                                continue
                            if not np.isinf(grid[i+kk[0], j+kk[1]]):
                                if nb_max < np.max(value_fn[i+kk[0], j+kk[1], :]):
                                    nb_max = np.max(value_fn[i+kk[0], j+kk[1], :])
                                    nb_dir = ii
                        if np.isinf(nb_max):    # all neighbours are walls
                            for kk in range(4):
                                policy[(j,i,kk)] = (0,0)
                        else:                   # turn around and move back
                            for kk in range(4):
                                if kk == nb_dir:
                                    policy[(j,i,kk)] = (1,0)
                                elif kk-nb_dir == 1 or kk-nb_dir == -3:
                                    policy[(j,i,kk)] = (0,-1)
                                else:
                                    policy[(j,i,kk)] = (0,1)
            # print(policy)
            # print(policy[(1,3,0)])
            # print(policy[(2,0,3)])
            # import pdb; pdb.set_trace()
            #print(policy)

            return policy

        def pass_wall(new_x, new_y, curr_x, curr_y):
            new_world_y, new_world_x = int(new_y/self.resolution), int(new_x/self.resolution)
            curr_world_y, curr_world_x = int(curr_y/self.resolution), int(curr_x/self.resolution)
            if self.collision_checker(new_world_x, new_world_y):
                return True     # is collision
            elif np.any(self.aug_map[min(new_world_y,curr_world_y):max(new_world_y,curr_world_y)+1, min(new_world_x,curr_world_x):max(new_world_x,curr_world_x)+1] > 0):
                return True    # is passing wall
            return False   # can walk through

        tmp_start = self.get_current_discrete_state()
        print(tmp_start)
        tmp_goal = self._get_goal_position()
        print(tmp_goal)

        grid = np.zeros((int(np.round(self.world_height*self.resolution)), int(np.round(self.world_width*self.resolution))))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if self.collision_checker(int(j/self.resolution), int(i/self.resolution)):
                    grid[i][j] = -np.inf
                # else:
                #     grid[i][j] = -1

        reward = np.zeros_like(grid)
        reward[:,:] = -0.4
        reward[tmp_goal[1], tmp_goal[0]] = 1.
        reward[grid[:,:] == -np.inf] = -2.

        # policy = value_iteration(grid)


        self.action_table = value_iteration(grid)


    def get_current_continuous_state(self):
        """Our state is defined to be the tuple (x,y,theta).
        x and y are directly extracted from the pose information.
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot
        """
        x = self.pose.pose.pose.position.x
        y = self.pose.pose.pose.position.y
        orientation = self.pose.pose.pose.orientation
        ori = [orientation.x, orientation.y, orientation.z,
               orientation.w]

        phi = np.arctan2(2 * (ori[0] * ori[1] + ori[2] * ori[3]), 1 - 2 *
                         (ori[1] ** 2 + ori[2] ** 2))
        return (x, y, phi)

    def get_current_discrete_state(self):
        """Our state is defined to be the tuple (x,y,theta).
        x and y are directly extracted from the pose information.
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot in discrete space, e.g., (1, 1, 1) where the robot is facing north
        """
        x, y, phi = self.get_current_continuous_state()
        def rd(x): return int(round(x))
        return rd(x), rd(y), rd(phi / (np.pi / 2))

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot

        Returns:
            bool -- True for collision, False for non-collision
        """
        if x >= self.world_width or x < 0 or y >= self.world_height or y < 0:
            return True
        if self.aug_map[y][x] > 0:
            return True
        return False

    def motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """predict the next pose of the robot given controls. Returns None if the robot collide with the wall
        The robot dynamics are provided in the homework description

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
            theta {float} -- current theta of robot
            v {float} -- linear velocity
            w {float} -- angular velocity

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        num_steps = int(dt * frequency)
        dx = 0
        dy = 0
        for i in range(num_steps):
            if w != 0:
                dx = - v / w * np.sin(theta) + v / w * \
                    np.sin(theta + w / frequency)
                dy = v / w * np.cos(theta) - v / w * \
                    np.cos(theta + w / frequency)
            else:
                dx = v*np.cos(theta)/frequency
                dy = v*np.sin(theta)/frequency
            x += dx
            y += dy

            if self.collision_checker(x, y):
                return None
            theta += w / frequency
        return x, y, theta

    def discrete_motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """discrete version of the motion predict. Note that since the ROS simulation interval is set to be 0.5 sec
        and the robot has a limited angular speed, to achieve 90 degree turns, we have to execute two discrete actions
        consecutively. This function wraps the discrete motion predict.

        Please use it for your discrete planner.

        Arguments:
            x {int} -- current x of robot
            y {int} -- current y of robot
            theta {int} -- current theta of robot
            v {int} -- linear velocity
            w {int} -- angular velocity (0, 1, 2, 3)

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        w_radian = w * np.pi/2
        first_step = self.motion_predict(x, y, theta*np.pi/2, v, w_radian)
        if first_step:
            second_step = self.motion_predict(
                first_step[0], first_step[1], first_step[2], v, w_radian)
            if second_step:
                return (round(second_step[0]), round(second_step[1]), round(second_step[2] / (np.pi / 2)) % 4)
        return None

    def publish_control(self):
        """publish the continuous controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1])
            self.controller.publish(msg)
            rospy.sleep(0.6)

    def publish_discrete_control(self):
        """publish the discrete controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(
                action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)

    def publish_stochastic_control(self):
        """publish stochastic controls in MDP.
        In MDP, we simulate the stochastic dynamics of the robot as described in the assignment description.
        Please use this function to publish your controls in task 3, MDP. DO NOT CHANGE THE PARAMETERS :)
        We will test your policy using the same function.
        """
        # current_state = self.get_current_state()
        current_state = self.get_current_discrete_state()
        import time

        actions = []
        new_state = current_state
        while not self._check_goal(current_state):
            # current_state = self.get_current_state()
            current_state = self.get_current_discrete_state()
            action = self.action_table[current_state[0],
                                       current_state[1], current_state[2] % 4]
            print("current state", current_state)
            print("action", action)
            if action == (1, 0):
                r = np.random.rand()
                if r < 0.9:
                    action = (1, 0)
                elif r < 0.95:
                    action = (np.pi/2, 1)
                else:
                    action = (np.pi/2, -1)
            print("Sending actions:", action[0], action[1]*np.pi/2)
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            # msg = create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            time.sleep(1)
            # current_state = self.get_current_state()
            current_state = self.get_current_discrete_state()


if __name__ == "__main__":
    # TODO: You can run the code using the code below
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    # TODO: You should change this value accordingly
    inflation_ratio = 3
    print("=================================")
    planner = Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    print("+++++++++++start+++++++++++++++")
    if planner.goal is not None:
        policy = planner.generate_plan()
    # import pdb; pdb.set_trace()

    print("+++++++++++saving+++++++++++++++")

    # for MDP, please dump your policy table into a json file
    
    dump_action_table(planner.action_table, "Controls/Task3_DSPA/DSPA_map1_"+str(goal[0])+"_"+str(goal[1])+".json")

    print("+++++++++++infer+++++++++++++++")
    # You could replace this with other control publishers
    #planner.publish_discrete_control()
    planner.publish_stochastic_control()

    print("+++++++++++end+++++++++++++++")



    # spin the ros
    rospy.spin()

