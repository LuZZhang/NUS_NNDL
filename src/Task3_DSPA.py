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
from base_planner import Planner as base_planner

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
    
   

class MDP_Planner(base_planner):
    
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
    
    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """

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
            #print("test0")
            while iter_flag and iterations <= max_iter:
                
                #print("iterations", iterations)
                value_fn_prev = np.copy(value_fn)
                iter_flag = False
                iterations += 1
                #print("grid.shape",grid.shape[0],grid.shape[1])
                flag = 0
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        # if i==1 and j == 1:
                            # import pdb; pdb.set_trace()
                            # print(value_fn_prev[0:3,:3,0])
                        for k in range(4):
                            if not np.isinf(grid[i][j]):
                                #flag = flag + 1
                                q0 = 0.
                                # a = (1, 0) #  forward
                                nbs = [(i + state_pos[k][0][0],j + state_pos[k][0][1],k + state_pos[k][0][2]), \
                                (i + state_pos[k][1][0],j + state_pos[k][1][1],k + state_pos[k][1][2]), \
                                (i + state_pos[k][2][0],j + state_pos[k][2][1],k + state_pos[k][2][2])] # f, # f_l, # f_r

                                for ii in range(3):
                                    # if (nbs[ii][1], nbs[ii][0], j, i)==(13, 19, 12, 18):
                                    #     import pdb; pdb.set_trace()
                                    if not check_posibility(nbs[ii][1], nbs[ii][0], j, i):
                                        # print(nbs[ii][1], nbs[ii][0], j, i)
                                        q0 += trns[ii] * (reward[nbs[ii][0]][nbs[ii][1]] + gamma * value_fn_prev[nbs[ii]])
                                        #flag +=1
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
                #print("flag:",flag)
            #print("test1")
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
                                if not check_posibility(nb[1], nb[0], j, i):
                                    action_values[a] = value_fn[nb]
                            if not action_values:
                                print("action_values = ",action_values)
                                #import pdb; pdb.set_trace()
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
            #print("test")
            return policy

        def check_posibility(new_x, new_y, curr_x, curr_y):
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
        flag = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if self.collision_checker(int(j/self.resolution), int(i/self.resolution)):
                    grid[i][j] = -np.inf
                    #flag = flag + 1
        #print("flag:",flag)            

        reward = np.zeros_like(grid)
        reward[:,:] = -0.4
        reward[tmp_goal[1], tmp_goal[0]] = 1.
        reward[grid[:,:] == -np.inf] = -2.
        #print("grid",grid)
        self.action_table = value_iteration(grid)
        
        #print("self.action_table :",self.action_table)



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
    inflation_ratio  = int(ceil((ROBOT_SIZE/resolution)))

    planner = MDP_Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])

    if planner.goal is not None:
        policy = planner.generate_plan()

    # for MDP, please dump your policy table into a json file
    dump_action_table(planner.action_table, "Controls/Task3_DSPA/DSPA_map1_"+str(goal[0])+"_"+str(goal[1])+".json")
    print("saving to Controls/Task3_DSPA/DSPA_map1_"+str(goal[0])+"_"+str(goal[1])+".json")
    
    # You could replace this with other control publishers
    #planner.publish_discrete_control()
    planner.publish_stochastic_control()

    # spin the ros
    rospy.spin()

