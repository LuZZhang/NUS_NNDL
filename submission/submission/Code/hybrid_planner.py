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
from scipy import ndimage
from std_msgs.msg import *
import heapq as hq
from geometry_msgs.msg import Pose
from base_planner import Planner as base_planner



ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit

output=0
class hybrid_astar(base_planner):
    
    def heuristic(self, position, target):
        #output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2)+(radians(position[2]) - radians(target[2])) ** 2)
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2))
        return float(output)

    def get_next(self, cont_state, neareast_reso=0.05, angular_reso = np.pi/4):
        nearest_reso = self.resolution
        x,y,theta = cont_state
        x = int(x/neareast_reso)*neareast_reso
        y = int(y/neareast_reso)*neareast_reso
        theta = int(theta/angular_reso) * angular_reso
        return [x, y, theta]
    
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
        
        
        
        self.action_seq = []
        
        cost_matrix = []
        v_step = 10
        w_step = 10
        for v_ in range(0, v_step):
            for w_ in range(0, w_step):
                v= v_ * (1.0/v_step)
                w = -np.pi + w_*2*np.pi/w_step
                cost_matrix.append([v,w,1])
        '''cost_matrix = [
                       [1,-pi,1],
                       [1,-pi,1],
                       [1,-pi/2,1],
                       [1,0,1] ,
                       [1,pi/2,1] ,
                       [1,pi,1]                                           
                       ]'''

        openheap=[] # would be grid position of nodes that is not explored, (grid_pos,cost)
        openheapCost={} # parent(grid_pos) - key and value as the cost of the exploration
        visitedNodes={} # parent(grid_pos) - key and value as the cost of the exploration
        dirNode={}

        posGoal=self._get_goal_position()
        xGoal=posGoal[0]
        yGoal=posGoal[1]
        tGoal=0
        
       
        start=self.get_current_continuous_state()
        posStart=self.get_next(start)
        xStart=posStart[0]
        yStart=posStart[1]
        tStart=posStart[2]

        startpos=(xStart,yStart,tStart)
        goalpos=(xGoal,yGoal,tGoal)        

        print("Starting Pos :", startpos , "  and Goal : " , goalpos )
        if(self.collision_checker(xGoal,yGoal) or self.collision_checker(xStart,yStart)):
            print(" Robot is start or Goal position near to obstacle, High chance of collision thus no path found !!!")
            return
        
        hq.heappush(openheap,(self.heuristic(startpos,goalpos),(startpos),[]))
        openheapCost[startpos]=(self.heuristic(startpos,goalpos),(startpos),[])
        final_path=[]
        
        while len(openheap)>0 :
            
            selectedNode=openheap[0] # Select the top node
            nodePos=selectedNode[1] # would be in x,y,t
            cost=selectedNode[0]  
            action_taken = selectedNode[2]       
            
            
            
            if self._check_goal(nodePos):

                    self.action_seq = visitedNodes[nodePos][2]
                    print("self.action_seq : " ,self.action_seq)
                    break
            
            hq.heappop(openheap)
            for i in cost_matrix:
                #print("exploring neighbour for the node : " , nodePos)
                # step cost from one node to its neighbour
                candidate_position = self.motion_predict(
                    x=nodePos[0] ,
                    y=nodePos[1] ,
                    theta=nodePos[2] ,
                    v=i[0],
                    w=i[1],
                )
       
                
                if candidate_position is None:
                    continue
                
                candidate_position = self.get_next(candidate_position)

                ngX=candidate_position[0]
                ngY=candidate_position[1]
                ngtheta = candidate_position[2] 

                ngNode=(ngX,ngY,ngtheta)

                if(ngX >= 0 and ngX < self.world_width and ngY >= 0 and ngY < self.world_height):
                    if(self.collision_checker(ngX,ngY)):
                        c=0
                    else:
                        gn= cost - self.heuristic(nodePos,goalpos)
                        total_cost= gn + self.heuristic(ngNode,goalpos) + i[2] #fn=gn+hn
                        flag=0
                        lo=0
                        if ngNode in openheapCost:
                            if total_cost >= openheapCost[ngNode][0]:
                                
                                #print(openheapCost[ngNode])
                                flag=1

                            elif ngNode in visitedNodes:
                                if total_cost >= visitedNodes[ngNode][0]:
                                    #print("Here : " ,visitedNodes ," \n")
                                    lo=1
                            

                        if flag==0 and lo==0:
                            
                            '''if ngNode[0] > 4:
                                print("(ngNode,total_cost) : ",ngNode,total_cost)'''
                            new_action_taken = action_taken + [[i[0],i[1]]]

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
    inflation_ratio = int(ceil((ROBOT_SIZE/resolution)))  # 1 is for safety
    inflation_ratio=5
    print("Inflation ratio : " , inflation_ratio )
    
    planner = hybrid_astar(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    planner.publish_control()

    # save your action sequence
    result = np.array(planner.action_seq)
    np.savetxt("2_maze3_9_4.txt", result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')
    print("Done")
    # spin the ros
    rospy.spin()
 
