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
from base_planner import Planner as base_planner

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit

class Hybrid_Astar(base_planner):
    
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
        print("---generate plan---")
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

        def check_posibility(new_x, new_y, curr_x, curr_y):
            new_world_y, new_world_x = int(new_y/self.resolution), int(new_x/self.resolution)
            curr_world_y, curr_world_x = int(curr_y/self.resolution), int(curr_x/self.resolution)
            current_map = []
            for i in range(len(self.aug_map)):
                current_map.append(self.aug_map[i])
                #print(self.aug_map[i])
            if self.collision_checker(new_x, new_y):
                return True     # is collision
            for i in range(min(new_world_y,curr_world_y),max(new_world_y,curr_world_y)+2):
                if(current_map[i]==100):
                    return True
            for i in range(min(new_world_x,curr_world_x),max(new_world_x,curr_world_x)+2):
                if(current_map[i]==100):
                    return True
            
            return False   # can walk through

        def get_neigbours(curr_cell):
            
            nb_coord = [(0, 1), (1, 0), (0, -1), (-1, 0)] # 4 neighbours: east, north, west, south
            curr_y, curr_x = curr_cell.coord
            curr_world_y, curr_world_x = int(curr_y/self.resolution), int(curr_x/self.resolution)
            curr_d = curr_cell.direction
            nbs = []
            for idx, p in enumerate(nb_coord):
                new_y, new_x = curr_y + p[0], curr_x + p[1]
                if not check_posibility(new_x, new_y, curr_x, curr_y):
                    new_cell = Cell()
                    new_cell.coord = (new_y, new_x)
                    new_cell.direction = idx
                    new_cell.parent = curr_cell
                    nbs.append(new_cell)
            return nbs
            
        def heuristic(x,g):
            return abs(x[0]-g[0])+abs(x[1]-g[1])
        
        def astar(start_state, goal_state):
            openset = [start_state]
            closeset = []
            #print("openset:",openset)
            while openset:
                idx_minf = np.argmin([cl.f for cl in openset])
                curr_state = openset.pop(idx_minf)
                closeset.append(curr_state)
                if curr_state.coord == goal_state.coord:
                    break
                # for nb in get_neigbours(curr_state):
                nnn = get_neigbours(curr_state)
                #print("nnn:",nnn)
                for nb in nnn:
                    if nb.coord in [cl.coord for cl in closeset]:
                        continue
                    
                    nb.g = curr_state.g + 1 if curr_state.direction == nb.direction else curr_state.g + 2
                    nb.h = heuristic(goal_state.coord,nb.coord)
                    
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
                    
            path = []
            while curr_state.parent:
                path.append(curr_state)
                curr_state = curr_state.parent
            path.append(curr_state)
            # print(path)
            return path[::-1]


        start_state = Cell()
        curr_start = self.get_current_continuous_state()
        start_state.coord = curr_start[0:2]
        start_state.direction = curr_start[2]
        goal_state = Cell()
        tmp_goal = self._get_goal_position()
        goal_state.coord = (tmp_goal[1], tmp_goal[0])
        print("start_state.coord , goal_state.coord :",start_state.coord,goal_state.coord)


        path = astar(start_state, goal_state)
        #print("len_path :",len(path))
        for i in range(1, len(path)):
            if path[i].direction - path[i-1].direction == 1 or path[i].direction - path[i-1].direction == -3:
                self.action_seq.append((0, np.pi)) # LEFT 
            elif path[i].direction - path[i-1].direction == -1 or path[i].direction - path[i-1].direction == 3:
                self.action_seq.append((0, -np.pi)) # RIGHT 
            self.action_seq.append((2, 0))

       
        print(self.action_seq)



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

    inflation_ratio = int(ceil((ROBOT_SIZE/resolution)))

    planner = Hybrid_Astar(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])

    if planner.goal is not None:
        planner.generate_plan()

    # save your action sequence
    result = np.array(planner.action_seq)
    #np.savetxt("Controls/Task2_CSDA/CSDA_com1_"+str(goal[0])+"_"+str(goal[1])+".txt", result, fmt="%.2e")
    txtname = "Controls/Task2_CSDA/CSDA_com1_"+ str(planner.goal.pose.position.x) + "_" +str(planner.goal.pose.position.y)+".txt"
    np.savetxt(txtname, result, fmt="%.2e")
    print("save to file:",txtname)

    # You could replace this with other control publishers

    planner.publish_control()

    # spin the ros
    rospy.spin()

