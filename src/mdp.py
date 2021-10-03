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
from std_msgs.msg import *
import heapq as hq
from base_planner import Planner as base_planner
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



class MarkovDecisionPlanner(base_planner):
    
    def __init__(self,world_width, world_height, world_resolution,inflation_ratio,actlist,transitions,states=None,gamma=0.95,delta=0.00001):
        if not (0<gamma<=1):
            raise ValueError("MDP must have gamma value between (0,1]")
        super(MarkovDecisionPlanner,self).__init__( world_width, world_height, world_resolution, inflation_ratio)
        self.states=states # would be forward(1) ,left(2) , right(3) , stay(4)
        self.transitions=transitions # [][]
        self.actlist=actlist # (1,0);(0,1);(0,-1);(0,0)
        self.gamma=gamma
        self.step_cost = -0.001
        self.goal_reward = 100
        self.collision_penalty = -10
        self.value_fn=np.zeros((int(world_width*resolution)+1,int(world_height*resolution)+1, len(self.states)))
        self.policy=np.zeros((int(world_width*resolution)+1,int(world_height*resolution)+1, len(self.states)))
        self.iteration_no=1
        self.delta=delta
        self.reward=np.zeros((int(world_width*resolution)+1,int(world_height*resolution)+1))


    def collision_checker(self, x, y):
        
        index = int(y/self.resolution)*self.world_width +int(x/self.resolution)
        if not (0<=y/self.resolution<self.world_height and 0<=x/self.resolution<self.world_width):
            return True
        if self.aug_map[index] == 100:
            return True
        else:
            
            return False

    def setReward(self):
        
        for i in range(int((self.world_width*self.resolution)+1)):
            for j in range(int((self.world_height*self.resolution)+1)):
                if self.collision_checker(i,j):
                    self.reward[i][j]=self.collision_penalty
                elif self._check_goal((i,j)):
                    self.reward[i][j]=self.goal_reward
                else:
                    self.reward[i][j]=self.step_cost
    def reset(self):
        self.value_fn=np.zeros((int(self.world_width*self.resolution)+1,int(self.world_height*self.resolution)+1, len(self.states)))
        self.policy=np.zeros((int(self.world_width*self.resolution)+1,int(self.world_height*self.resolution)+1, len(self.states)))
        self.iteration_no=1
        self.setReward()

    def getReward(self,i,j):
        return self.reward[i][j]
        
    def getTransitionProb(self,state,action):
        return self.transitions[state][action]
    
    def getValueFn(self,fn,state,k):
        if not state:
            return self.collision_penalty
        x=state[0]
        y=state[1]
        return fn[int(x)][int(y)][int(k)]

    def generate_plan(self):
        # Publish inflated map in a topic
        test_map = OccupancyGrid()
        test_map.info.resolution = self.resolution
        test_map.info.width = self.world_width
        test_map.info.height = self.world_height
        test_map.info.origin.position.x = 0.0 
        test_map.info.origin.position.y = 0.0 
        test_map.info.origin.position.z = 0.0 
        test_map.info.origin.orientation.x = 0.0 
        test_map.info.origin.orientation.y = 0.0 
        test_map.info.origin.orientation.z = 0.0 
        test_map.info.origin.orientation.w = 0.0 
        for i in range(self.world_width*self.world_height):
            test_map.data.append(self.aug_map[i])
        #test_map.data = self.aug_map
        map_pub = rospy.Publisher('/map_inf', OccupancyGrid,latch=True,queue_size=10)
        map_pub.publish(test_map)

        self.reset()
        base_fn=np.zeros((int(self.world_width*self.resolution)+1,int(self.world_height*self.resolution)+1, len(self.states)))
        flag=1
        while True:
            print("epcho: " , self.iteration_no)
            flag=0
            self.iteration_no+=1
            for curr_x in range(int(self.world_width*self.resolution)):
                for curr_y in range(int(self.world_height*self.resolution)):
                    
                    for angle in [0,1,2,3]:
                        exp_ret = []
                        if self._check_goal((curr_x,curr_y)):
                            #print("Goal reached")
                            self.value_fn[curr_x][curr_y][angle]= self.goal_reward
                            
                            #print("if condition of goal : " , self.value_fn)
                        else:
                            if self.collision_checker(curr_x, curr_y):
                                #print("collision")
                                self.value_fn[curr_x][curr_y][angle]= self.collision_penalty
                                continue
                            for i in range(0,len(self.actlist)):
                                #For probabilitic action
                                #print("states : " , self.states[i])
                                
                                if self.states[i]==1:
                                    nextForwardState=self.discrete_motion_predict(curr_x,curr_y,angle, 1,0)
                                    nextLeftState=self.discrete_motion_predict(curr_x,curr_y,angle, np.pi/2,1)
                                    nextRightState=self.discrete_motion_predict(curr_x,curr_y,angle, np.pi/2,-1)
                                    #print("nextForwardState : ", nextForwardState)
                                    
                                    #print("self.getValueFn(nextForwardState,i) : ", self.getValueFn(base_fn,nextForwardState,i))
                                    ret=self.getTransitionProb(0,0)* self.getValueFn(base_fn,nextForwardState,i)+ \
                                        self.getTransitionProb(0,1)* self.getValueFn(base_fn,nextLeftState,i) + \
                                        self.getTransitionProb(0,2) * self.getValueFn(base_fn,nextRightState,i)
                                    exp_ret.append(ret)
                                    
                                else:
                                    # For determintic action
                                    
                                    nextstate =  self.discrete_motion_predict(curr_x,curr_y,angle, 0, actlist[i][1])
                                    if nextstate==None:
                                        continue
                                    #print("nextstate : " ,nextstate)
                                    nextX, nextY, nextAngle=nextstate
                                    exp_ret.append(base_fn[int(nextX)][int(nextY)][int(nextAngle)])
                                    
                            
                            self.policy[curr_x][curr_y][angle] = np.argmax(exp_ret)
                            comp=self.step_cost + self.gamma * max(exp_ret)
                            self.value_fn[curr_x][curr_y][angle]=self.step_cost + self.gamma * max(exp_ret)
            
            if np.max(np.abs(self.value_fn - base_fn)) > self.delta:
                flag=1
            
           
            base_fn=self.value_fn.copy()
        
            if flag ==0:
                break
        
        #print(self.policy)

        for x in range(int(self.world_width*self.resolution)):
            for y in range(int(self.world_height*self.resolution)):
                for w in [0,1,2,3]:
                    assert self.policy[x][y][w] == int(self.policy[x][y][w])
                    
                    self.action_table[(x,y,w)] = self.actlist[int(self.policy[x][y][w])]



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
    inflation_ratio = int(ceil((ROBOT_SIZE/resolution)))  # 1 is for safety
    print("Inflation ratio : " , inflation_ratio)
    
    actlist=[[1,0],[0,1],[0,-1]]
    transitions=[[0.9 , 0.05 ,0.05,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    states=[1,2,3,4]
    planner = MarkovDecisionPlanner(width, height, resolution, inflation_ratio=inflation_ratio,actlist=actlist,transitions=transitions,states=states)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # save your action sequence
    #result = np.array(planner.action_seq)
    #np.savetxt("actions_continuous.txt", result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    txtname = "Controls/Task3_DSPA/DSPA_com1_"+ str(planner.goal.pose.position.x) + "_" +str(planner.goal.pose.position.y)+".json"
    dump_action_table(planner.action_table, txtname)
    print("saving to ",txtname)
    planner.publish_stochastic_control()
    print("Done")

    # spin the ros
    rospy.spin()
 
