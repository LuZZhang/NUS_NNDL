#!/usr/bin/env python
import rospy
import numpy as np
import time
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse
from base_planner import Planner as base_planner
import heapq as hq
ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit



class AstarPlanner(base_planner):
    

    def check_posibility(self,x,y,direction):
        total_dis = 1 #Distance travelled by robot in pixels
        total_step = int(total_dis/self.resolution) #total steps wrt to resolution
        
        for step_count in range(0,total_step+1 ,1):
            step_diff = step_count * self.resolution #the distance travelled 
            step_location = [(x+step_diff,y), (x,y+step_diff), (x-step_diff,y), (x,y-step_diff)]
            if self.collision_checker(step_location[direction][0], step_location[direction][1]):
                # if there is collision, then cannot move forward
                return False

        return True

    def heuristic(self,x,g):

        return abs(x[0]-g[0])+abs(x[1]-g[1])


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
        possible_neighbour=[[1,0],[0,1],[0,-1]]
        cost_updates =[0.8,1,1.2] # Prefer to move more straight when the cost of turning and straight is same
        openheap=[] # would be grid position of nodes that is not explored, (grid_pos,cost)
        openheapCost={} # parent(grid_pos) - key and value as the cost of the exploration
        visitedNodes={} # parent(grid_pos) - key and value as the cost of the exploration
        dirNode={}

        posStart=self.get_current_discrete_state()
        xStart = posStart[0] #int(ceil(posStart[0]/self.resolution))
        yStart = posStart[1] #int(ceil(posStart[1]/self.resolution))
        tStart = posStart[2]

        posGoal=self._get_goal_position()
        xGoal= posGoal[0] #int(ceil(posGoal[0]/self.resolution))
        yGoal=posGoal[1]#int(ceil(posGoal[1]/self.resolution))
        tGoal=0

        startpos=(xStart,yStart,tStart)
        goalpos=(xGoal,yGoal,tGoal)

        print("Starting Pos :", startpos , "  and Goal : " , goalpos )
        if(self.collision_checker(startpos[0],startpos[1]) or self.collision_checker(goalpos[0],goalpos[1])):
            print(" Robot is start or Goal position near to obstacle, High chance of collision thus no path found !!!")
            return

        hq.heappush(openheap,(self.heuristic(startpos,goalpos),(startpos),[]))
        openheapCost[startpos]=(self.heuristic(startpos,goalpos),(startpos),[])
        final_path=[]

        while len(openheap)>0:
            
            selectedNode=openheap[0] # Select the top node
            
            nodePos=selectedNode[1] # would be in x,y, theta
            cost=selectedNode[0]
            action_taken = selectedNode[2]

            #visitedNodes[nodePos]=openheapCost[nodePos]
            
            # if goal is reached
            if goalpos in visitedNodes:
                self.action_seq = visitedNodes[goalpos][2]
                print("self.action_seq : " ,self.action_seq)
                break
            
            #print(" openheap Current : " , openheap[0])
            hq.heappop(openheap)
            for i in range(0,3):
                
                #print("exploring neighbour for the node : " , nodePos)
                gn = cost - self.heuristic(nodePos,goalpos)

                x = nodePos[0]
                y = nodePos[1]
                theta = nodePos[2]
                
                if possible_neighbour[i] == [1,0]:
                    # moves forward  
                    if(not (self.check_posibility(x,y,theta) )):
                        continue                
                    next_locations = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
                    new_x, new_y = next_locations[theta]  # use theta as an index
                    ngX,ngY,ngTheta = (new_x, new_y, theta) # theta is not changed

                if possible_neighbour[i] == [0,1]:
                    # left 90 degrees
                    ngX,ngY,ngTheta = (x, y , (theta+1)%4)
                if possible_neighbour[i] == [0,-1]:
                    ngX,ngY,ngTheta = (x,y, (theta-1)%4)              

                ngNode=(ngX,ngY,ngTheta)

                
                if(ngX >= 0 and ngX < self.world_width and ngY >= 0 and ngY < self.world_height):
                    if(self.collision_checker(ngX,ngY)):

                        continue
                    else:
                        total_cost=gn+1+self.heuristic(ngNode,goalpos)#+cost_updates[i] #fn=gn+hn
                        flag=0
                        lo=0
                        if ngNode in openheapCost:
                            if total_cost > openheapCost[ngNode][0]:
                                flag=1
                            elif ngNode in visitedNodes:
                                
                                if total_cost > visitedNodes[ngNode][0]:
                                    #print("Here : " ,visitedNodes ," \n")
                                    lo=1

                        if flag==0 and lo==0:
                            
                
                            new_action_taken = action_taken + [possible_neighbour[i],]
                            #print("(ngNode,total_cost) : ",ngNode,total_cost,"   new_action_taken : " , new_action_taken)
                            
                            hq.heappush(openheap,(total_cost,ngNode,new_action_taken))
                            openheapCost[ngNode]=(total_cost,selectedNode,new_action_taken)
                            visitedNodes[ngNode]=openheapCost[ngNode]
                         
            
                
            



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
    inflation_ratio = int(ceil((ROBOT_SIZE/resolution))) #ori is 3
    #print("inflation ratio = ",inflation_ratio)
    
    planner = AstarPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # save your action sequence
    result = np.array(planner.action_seq)
    txtname = "Controls/DSDA_map2_"+ str(planner.goal.pose.position.x) + "_" +str(planner.goal.pose.position.y)+".txt"
    np.savetxt(txtname, result, fmt="%.2e")
    print("save to file:",txtname)
    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')
    
    # You could replace this with other control publishers
    planner.publish_discrete_control()
    # spin the ros
    rospy.spin()
