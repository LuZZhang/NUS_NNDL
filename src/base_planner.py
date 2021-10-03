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
import time

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


class Planner(object):
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
        self.grid=np.zeros((self.world_width,self.world_height))
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
        #print(self.map)
        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask

        # you should inflate the map to get self.aug_map
        #self.aug_map = copy.deepcopy(self.map)
        self.aug_map=np.copy(self.map)
        print("Map call back")
        old_grid=np.zeros((self.world_width,self.world_height))
        for i in range(self.world_width):
            for j in range(self.world_height):
                self.grid[i][j]=self.aug_map[j*self.world_width+i]
                old_grid[i][j]=self.grid[i][j]

        flag=1 
        for i in range(self.world_width):
            for j in range(self.world_height):
                if(old_grid[i][j]==100):
                    flag1=0
                    for k in range(int(self.inflation_ratio)): 
                        if (j+k < self.world_height):
                            self.grid[i][j+k]=100
                            #flag1 = flag1 + 1 
                        if(j-k > 0):
                            self.grid[i][j-k]=100
                            #flag1 = flag1 + 1 
                        if(i+k < self.world_width):
                            self.grid[i+k][j]=100 
                            #flag1 = flag1 + 1                            
                        if(i-k > 0):
                            self.grid[i-k][j]=100
                            #flag1 = flag1 + 1
                    #print("i,j,flag:",i,j,flag1)             
                else:
                    continue
        #print("int(self.inflation_ratio) : ", int(self.inflation_ratio))
        zero_num = 0
        for i in range(self.world_width):
            for j in range(self.world_height):
                self.aug_map[j*self.world_width+i]=self.grid[i][j]
                #if(self.grid[i][j]==100):
                    #flag = flag + 1
                #else:
                    #zero_num = zero_num + 1
        #print("flag:",flag)
        #print("zeros:",zero_num)
        #print("Size : " , self.aug_map.shape,self.world_width,self.world_height )
        #for i in range(self.world_width):
            #print(self.aug_map[i*self.world_height:(i+1)*self.world_height])

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
        if self._d_from_goal(pose) < 0.5:
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



    def heuristic (self,x,g):

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
        # Publish inflated map in a topic
        print("start to generate plan")
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
        i=int(ceil(x/self.resolution))
        j=int(ceil(y/self.resolution))
        index = j*self.world_width +i
        if i <= 0 or i >= ceil(self.world_width - 1):
            return True
        elif j <= 0 or j >= ceil(self.world_height - 1):
            return True
        if(self.aug_map[index]==100):
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
            rospy.sleep(2)

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
        current_state = self.get_current_discrete_state()
        actions = []
        new_state = current_state
        while not self._check_goal(current_state):
            current_state = self.get_current_discrete_state()
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
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            time.sleep(1)
            current_state = self.get_current_discrete_state()

        self.action_seq = actions


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

    planner = Planner(width, height, resolution, inflation_ratio=inflation_ratio)   
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    #planner.publish_discrete_control()

    # save your action sequence
    result = np.array(planner.action_seq)
    txtname = "Controls/DSDA_com1_"+ str(planner.goal.pose.position.x) + "_" +str(planner.goal.pose.position.y)+".txt"
    np.savetxt(txtname, result, fmt="%.2e")
    print("save to file:",txtname)
    #np.savetxt("task_1.txt", result, fmt="%.2e")

    # You could replace this with other control publishers
    planner.publish_discrete_control()
    
    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')
    print("Done")
    # spin the ros
    rospy.spin()
 
