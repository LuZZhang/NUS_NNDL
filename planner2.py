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

        Each action could be: (1, 0) FORWARD, (0, np.pi) LEFT 90 degree, (0, -np.pi) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """
        self.action_seq = []

        class Cell:
            def __init__(self):
                self.coord = (0, 0)
                self.direction = 0
                self.parent = None
                self.g = 0
                self.h = 0
                self.f = self.g + self.h

            def __eq__(self, cell):
                return self.coord == cell.coord

        def pass_wall(new_x, new_y, curr_x, curr_y):
            new_world_y, new_world_x = int(new_y/self.resolution), int(new_x/self.resolution)
            curr_world_y, curr_world_x = int(curr_y/self.resolution), int(curr_x/self.resolution)
            if self.collision_checker(new_world_x, new_world_y):
                return True     # is collision
            elif np.any(self.aug_map[min(new_world_y,curr_world_y):max(new_world_y,curr_world_y)+1, min(new_world_x,curr_world_x):max(new_world_x,curr_world_x)+1] > 0):
                return True    # is passing wall
            return False   # can walk through

        def get_neigbours(curr_cell):
            # nb_coord = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 8 neighbours
            nb_coord = [(0, 1), (1, 0), (0, -1), (-1, 0)] # 4 neighbours: east, north, west, south
            curr_y, curr_x = curr_cell.coord
            curr_world_y, curr_world_x = int(curr_y/self.resolution), int(curr_x/self.resolution)
            curr_d = curr_cell.direction
            nbs = []
            for idx, p in enumerate(nb_coord):
                new_y, new_x = curr_y + p[0], curr_x + p[1]
                if not pass_wall(new_x, new_y, curr_x, curr_y):
                    new_cell = Cell()
                    new_cell.coord = (new_y, new_x)
                    new_cell.direction = idx
                    new_cell.parent = curr_cell
                    nbs.append(new_cell)
            return nbs

        def astar(start_state, goal_state):
            openset = [start_state]
            closeset = []

            while openset:
                idx_minf = np.argmin([cl.f for cl in openset])
                curr_state = openset.pop(idx_minf)
                closeset.append(curr_state)
                # print("curr_state", curr_state.coord)
                # import pdb
                # pdb.set_trace()
                if curr_state.coord == goal_state.coord:
                    break
                # for nb in get_neigbours(curr_state):
                nnn = get_neigbours(curr_state)
                for nb in nnn:
                    # print("nb", nb.coord)
                    # import pdb
                    # pdb.set_trace()
                    if nb.coord in [cl.coord for cl in closeset]:
                        continue
                    #nb.g = curr_state.g + 1 # rotation don't cost
                    nb.g = curr_state.g + 1 if curr_state.direction == nb.direction else curr_state.g + 2
                    nb.h = abs(goal_state.coord[0] - nb.coord[0]) + abs(goal_state.coord[1] - nb.coord[1]) # manhattan dist
                    #nb.h = (goal_state.coord[0] - nb.coord[0]) ** 2 + (goal_state.coord[1] - nb.coord[1]) ** 2
                    found = False
                    for i in range(len(openset)):
                        if openset[i].coord == nb.coord:
                            found = True
                            if openset[i].f > nb.f:
                                del openset[i]
                                openset.append(nb)
                            break
                    if not found:
                        openset.append(nb)
                    # print("add nb to openset", nb.coord)
            path = []
            while curr_state.parent:
                path.append(curr_state)
                curr_state = curr_state.parent
            path.append(curr_state)
            # print(path)
            return path[::-1]


        start_state = Cell()
        tmp_start = self.get_current_continuous_state()
        print(tmp_start)
        start_state.coord = tmp_start[0:2]
        start_state.direction = tmp_start[2]
        goal_state = Cell()
        tmp_goal = self._get_goal_position()
        goal_state.coord = (tmp_goal[1], tmp_goal[0])
        print(goal_state.coord)


        path = astar(start_state, goal_state)
        for i in range(1, len(path)):
            if path[i].direction - path[i-1].direction == 1 or path[i].direction - path[i-1].direction == -3:
                self.action_seq.append((0, np.pi)) # LEFT 90 degree
            elif path[i].direction - path[i-1].direction == -1 or path[i].direction - path[i-1].direction == 3:
                self.action_seq.append((0, -np.pi)) # RIGHT 90 degree
            self.action_seq.append((2, 0))

        # self.action_seq = [(0,np.pi),(0,np.pi),(2,0),(1,0)]
        print(self.action_seq)




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
        current_state = self.get_current_state()
        actions = []
        new_state = current_state
        while not self._check_goal(current_state):
            current_state = self.get_current_state()
            action = self.action_table[current_state[0],
                                       current_state[1], current_state[2] % 4]
            if action == (1, 0):
                r = np.random.rand()
                if r < 0.9:
                    action = (1, 0)
                elif r < 0.95:
                    action = (np.pi/2, 1)
                else:
                    action = (np.pi/2, -1)
            print("Sending actions:", action[0], action[1]*np.pi/2)
            msg = create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            time.sleep(1)
            current_state = self.get_current_state()


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
        planner.generate_plan()
    # import pdb; pdb.set_trace()

    print("+++++++++++saving+++++++++++++++")

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, "results/1_com1_"+str(goal[0])+"_"+str(goal[1])+".json")

    # save your action sequence
    file = open("results/2_maze3_"+str(goal[0])+"_"+str(goal[1])+".txt", "a")
    result = np.array(planner.action_seq)
    np.savetxt("results/2_maze3_"+str(goal[0])+"_"+str(goal[1])+".txt", result, fmt="%.2e")

    print("+++++++++++infer+++++++++++++++")
    # You could replace this with other control publishers
    # planner.publish_discrete_control()
    planner.publish_control()
    # planner.publish_stochastic_control()

    print("+++++++++++end+++++++++++++++")

    # spin the ros
    rospy.spin()
