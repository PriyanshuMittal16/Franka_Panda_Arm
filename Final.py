import sys
import numpy as np
from copy import deepcopy
from math import pi,sin,cos
from lib.astar import Astar
#from lib.loadmap import loadmap

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from lib.calculateIK6 import IK
#from lib.calculateIK6 import IK
#from lib.solverIK import IKd
from lib.calculateFK import FK
#from lib.solveIK import IK

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
ik=IK()
fk=FK()

def move_arm(goal,start):
	
    print("goal", goal, "start", start)
    goal[2][3]=goal[2][3]+0.01
    q = ik.inverse(start,goal)
    print(q)
    arm.safe_move_to_position(q)   
    return q

def pick_and_place(team, target, H_ee,i):
    
    if team == 'red':
      #start=arm.neutral_position()
      #start = np.array([-1.37265 , 1.46383 , 1.99189 ,-1.76744 ,-1.37034 , 1.183  ,  1.24244])
      start = np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
      #start = np.array([-3.11176606e-01 ,-3.57905471e-02 ,-7.56097364e-03 ,-1.90897320e+00, -2.68731754e-04 , 1.86965151e+00 , 4.66735091e-01])
      start3 =np.array([-0.28828845 , 0.37559731 ,-0.00941477, -0.46700214 , 0.00466382 , 0.86561005, 0.49145471])
      #arm.safe_move_to_position(start)
      #start = np.array([-1.99,-1.73,-1.49,-1.66,-0.47,3.14,-1.78])
      
      jointposition, T0e= fk.forward(start)
      g = np.matmul(T0e, H_ee)
      goal = np.matmul(g, target)
      goal_rot = goal[0:3,0:3]
      goal_rot_c1 = goal_rot[:,0]
      goal_rot_c2 = goal_rot[:,1]
      goal_rot_c3 = goal_rot[:,2]
      base_z = np.array([0,0,1])
      dot_c1 = np.dot(goal_rot_c1,base_z)
      dot_c2 = np.dot(goal_rot_c2,base_z)
      dot_c3 = np.dot(goal_rot_c3,base_z)
      
      if (dot_c1>0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp3,temp2,-temp1))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
      if (dot_c1<-0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp2,temp3,temp1))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
      if (dot_c2>0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp1,temp3,-temp2))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
        
      if (dot_c2<-0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp3,temp1,temp2))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
        
      if (dot_c3>0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp2,temp1,-temp3))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
      a=goal[0][3]
      b=goal[1][3]
      c=goal[2][3]
      
      block_dash=np.array([[7.07108437e-01,  -7.07105126e-01,  2.91009295e-06,
          a],
        [ -7.07105126e-01, -7.07108437e-01,  -2.63579713e-06,
          b],
        [ 3.92153693e-06,  -1.93947254e-07, -1.00000000e+00,
          c],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00]])
                            
      goal1 = np.matmul(goal, block_dash)
      
      q = move_arm(goal,start)

      

      #Going directly to the block. We can add the functionality that go above it first and then come down
      arm.exec_gripper_cmd(0.049, 15)  #Change force value as required
      arm.safe_move_to_position(start)  #decide where to move back: neutral position or hard code some position
      
      
      
      mid_final = np.array([[1,  0,  0,
          0.561],
        [ 0, -1,  -0,
          0.169],
        [ 0,  0, -1,
          0.6],
        [ 0,  0  ,0,
          1]])
      
      #for i in range(4):
      #q = move_arm(mid_final, start1)
      mid_f=np.array([0.4750 , 0.2306 ,  -0.2339,  -1.0616,  0.0552 ,  1.2869, 1.8334])
      arm.safe_move_to_position(mid_f)
      
      final = np.array([[7.07108437e-01,  -7.07105126e-01,  2.91009295e-06,
          0.561],
        [ -7.07105126e-01, -7.07108437e-01,  -2.63579713e-06,
          .169],
        [ 3.92153693e-06,  -1.93947254e-07, -1.00000000e+00,
          0.25+i*0.05],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00]])
      
      #q = move_arm(final, q)
      
      if(i==0):
        f=np.array([ 0.48949409,  0.21616586, -0.20518379, -2.03049792 , 0.05587792 , 2.24120487, 1.82498079 ])
      elif(i==1):
        f=np.array([ 0.4925436 ,  0.1549129 , -0.20861176 ,-1.97503214 , 0.03767253 , 2.12600892, 1.83726105])
      elif(i==2):
        f=np.array([ 0.49485142 , 0.11066599, -0.21053017 ,-1.8995811  , 0.02551874 , 2.00737784, 1.84554967])
      elif(i==3):
        f=np.array([ 0.49679362 , 0.08459057, -0.2122611 , -1.80314792 , 0.01875806 , 1.88546528, 1.85023589])
      arm.safe_move_to_position(f)
      arm.exec_gripper_cmd(0.15, 15)
      arm.safe_move_to_position(mid_f)
      if(i==0 or i==1 or i==2):
        arm.safe_move_to_position(start)
      else:
        new_dynamic=np.array([ 1.55884611 , 0.96035708 , 0.19518399 ,-0.08630148, -0.18476489 , 1.04486523, 2.36358211])
        arm.safe_move_to_position(new_dynamic)
    
    else:
      #start=arm.neutral_position()
      #start = np.array([-1.37265 , 1.46383 , 1.99189 ,-1.76744 ,-1.37034 , 1.183  ,  1.24244])
      #start1 = np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
      start = np.array([ 0.22989307 , 0.24263811 , 0.08206602, -1.03253644, -0.02059894 ,  1.27409853, 1.08894914]) 
      start3 =np.array([-0.28828845 , 0.37559731 ,-0.00941477, -0.46700214 , 0.00466382 , 0.86561005, 0.49145471])
      #start=np.array([0.4750 , 0.2306 ,  -0.2339,  -1.0616,  0.0552 ,  1.2869, 1.8334])
      #arm.safe_move_to_position(start)
      #start = np.array([-1.99,-1.73,-1.49,-1.66,-0.47,3.14,-1.78])
      
      jointposition, T0e= fk.forward(start)
      g = np.matmul(T0e, H_ee)
      goal = np.matmul(g, target)
      goal_rot = goal[0:3,0:3]
      goal_rot_c1 = goal_rot[:,0]
      goal_rot_c2 = goal_rot[:,1]
      goal_rot_c3 = goal_rot[:,2]
      base_z = np.array([0,0,1])
      dot_c1 = np.dot(goal_rot_c1,base_z)
      dot_c2 = np.dot(goal_rot_c2,base_z)
      dot_c3 = np.dot(goal_rot_c3,base_z)
      
      if (dot_c1>0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp3,temp2,-temp1))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
      if (dot_c1<-0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp2,temp3,temp1))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
      if (dot_c2>0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp1,temp3,-temp2))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
        
      if (dot_c2<-0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp3,temp1,temp2))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
        
      if (dot_c3>0.9):
        temp1 = np.array([goal[0,0],goal[1,0],goal[2,0]])
        temp2 = np.array([goal[0,1],goal[1,1],goal[2,1]])
        temp3 = np.array([goal[0,2],goal[1,2],goal[2,2]])
        A = np.column_stack((temp2,temp1,-temp3))
        A = np.column_stack((A,np.array([goal[0,3],goal[1,3],goal[2,3]])))
        goal = np.row_stack((A,np.array([goal[3,0],goal[3,1],goal[3,2],goal[3,3]])))
        
      a=goal[0][3]
      b=goal[1][3]
      c=goal[2][3]
      
      block_dash=np.array([[7.07108437e-01,  -7.07105126e-01,  2.91009295e-06,
          a],
        [ -7.07105126e-01, -7.07108437e-01,  -2.63579713e-06,
          b],
        [ 3.92153693e-06,  -1.93947254e-07, -1.00000000e+00,
          c],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00]])
                            
      goal1 = np.matmul(goal, block_dash)
      
      q = move_arm(goal,start)

      

      #Going directly to the block. We can add the functionality that go above it first and then come down
      arm.exec_gripper_cmd(0.049, 15)  #Change force value as required
      arm.safe_move_to_position(start)  #decide where to move back: neutral position or hard code some position
      
      
      
      mid_final = np.array([[1,  0,  0,
          0.561],
        [ 0, -1,  -0,
          0.169],
        [ 0,  0, -1,
          0.6],
        [ 0,  0  ,0,
          1]])
      
      #for i in range(4):
      #q = move_arm(mid_final, start1)
      mid_f= np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
      arm.safe_move_to_position(mid_f)
      
      final = np.array([[7.07108437e-01,  -7.07105126e-01,  2.91009295e-06,
          0.561],
        [ -7.07105126e-01, -7.07108437e-01,  -2.63579713e-06,
          .169],
        [ 3.92153693e-06,  -1.93947254e-07, -1.00000000e+00,
          0.25+i*0.05],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00]])
      
      #q = move_arm(final, q)
      
      if(i==0):
        f=np.array([ 0.10725433 , 0.23124332 ,-0.41759178 ,-2.02939213 , 0.11872247 , 2.23766118, 1.19665728 ])
      elif(i==1):
        f=np.array([ 0.12802434 , 0.16724282, -0.43988585 ,-1.97446405 , 0.08354272 , 2.12418056, 1.22028172])
      elif(i==2):
        f=np.array([ 0.14633045 , 0.12047868, -0.45799492 ,-1.89929451 , 0.05875013 , 2.00648997, 1.23713742])
      elif(i==3):
        f=np.array([ 0.16285832 , 0.09276819 ,-0.47463442 ,-1.80306906 , 0.04461067 , 1.885022, 1.24692366])
      arm.safe_move_to_position(f)
      arm.exec_gripper_cmd(0.15, 15)
      arm.safe_move_to_position(mid_f)
      if(i==0 or i==1 or i==2):
        arm.safe_move_to_position(start3)
      else:
        new_dynamic=np.array([0.74938978, -0.88994275, -1.71396112 ,-1.25485352 ,-0.8824145 ,  1.48073024, 1.20047301])
        arm.safe_move_to_position(new_dynamic)

      return q

def pick_and_place_dynamic(team,  H_ee,j):
  if team=='red':
    H_block=np.array([[0,0,-1,0.12], [0,-1,0,0.72], [-1,0,0,0.24], [0,0,0,1]]) #array to pick up blocks from
  #start=arm.neutral_position()
    #start = np.array([-1.37265 , 1.46383 , 1.99189 ,-1.76744 ,-1.37034 , 1.183  ,  1.24244])
    #start = np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
    #start = np.array([-3.11176606e-01 ,-3.57905471e-02 ,-7.56097364e-03 ,-1.90897320e+00, -2.68731754e-04 , 1.86965151e+00 , 4.66735091e-01])
    #arm.safe_move_to_position(start)
    #start = np.array([-1.99,-1.73,-1.49,-1.66,-0.47,3.14,-1.78])
    start=np.array([   1.43546162 , 0.78929772 , 0.20967993, -1.1674336 , -0.1591745  , 1.94108375, 2.42770667])
    arm.exec_gripper_cmd(0.1, 25)
   
    #start = np.array([0.712, 1.07, 0.5257, -1.144, 0.886, 1.305, 1.82])
    arm.safe_move_to_position(start)
    arm.exec_gripper_cmd(0.045, 25)
    state = arm.get_gripper_state()
    dforce=state['force']
    #print(dforce)
    
    
    #pos = state['position']
    #length=abs(pos[0]+pos[1])
   
    while (dforce[0]<10):
      arm.exec_gripper_cmd(0.1, 50)
      arm.exec_gripper_cmd(0.049, 50)
      state = arm.get_gripper_state()
      dforce=state['force']
      
    
    new_dynamic=np.array([ 1.55884611 , 0.96035708 , 0.19518399 ,-0.08630148, -0.18476489 , 1.04486523, 2.36358211])
        
    arm.safe_move_to_position(new_dynamic)
    mid_f=np.array([0.4750 , 0.2306 ,  -0.2339,  -1.0616,  0.0552 ,  1.2869, 1.8334])
    arm.safe_move_to_position(mid_f)

    if (j==1):
      f=np.array([0.49805785 , 0.07796541 ,-0.21467503 ,-1.68368044 , 0.01692403 , 1.75949897, 1.851615474])
    elif (j==2):
      f=np.array([0.4975769  , 0.09331004 ,-0.21845678 ,-1.53633061 , 0.02025225 , 1.62707648, 1.84967721])


    arm.safe_move_to_position(f)
    arm.safe_move_to_position(mid_f)
    arm.safe_move_to_position(new_dynamic)
  ###############################################################################Dynamic############################################################
  else:
    H_block=np.array([[0,0,-1,0.12], [0,-1,0,0.72], [-1,0,0,0.24], [0,0,0,1]]) #array to pick up blocks from
  #start=arm.neutral_position()
    #start = np.array([-1.37265 , 1.46383 , 1.99189 ,-1.76744 ,-1.37034 , 1.183  ,  1.24244])
    #start = np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
    #start = np.array([-3.11176606e-01 ,-3.57905471e-02 ,-7.56097364e-03 ,-1.90897320e+00, -2.68731754e-04 , 1.86965151e+00 , 4.66735091e-01])
    #arm.safe_move_to_position(start)
    #start = np.array([-1.99,-1.73,-1.49,-1.66,-0.47,3.14,-1.78])
    start=np.array([   0.91643413, -1.53079082 ,-1.3649686  ,-1.10494128, -1.62806907 , 1.36921594, -0.33646106])
    arm.exec_gripper_cmd(0.1, 25)
   
    #start = np.array([0.712, 1.07, 0.5257, -1.144, 0.886, 1.305, 1.82])
    arm.safe_move_to_position(start)
    arm.exec_gripper_cmd(0.045, 25)
    state = arm.get_gripper_state()
    dforce=state['force']
    #print(dforce)
    
    
    #pos = state['position']
    #length=abs(pos[0]+pos[1])
   
    while (dforce[0]<10):
      arm.exec_gripper_cmd(0.1, 50)
      arm.exec_gripper_cmd(0.049, 50)
      state = arm.get_gripper_state()
      dforce=state['force']
      
    
    new_dynamic=np.array([0.74938978, -0.88994275, -1.71396112 ,-1.25485352 ,-0.8824145 ,  1.48073024, 1.20047301])
        
    arm.safe_move_to_position(new_dynamic)
    mid_f= np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
    arm.safe_move_to_position(mid_f)

    if (j==0):
      f=np.array([0.17676222 , 0.08617009, -0.49193402, -1.683575   , 0.04146101 , 1.75908268, 1.24935926])
    elif (j==1):
      f=np.array([0.18520347 , 0.1037564 , -0.51165627 ,-1.53622743 , 0.05086541 , 1.62636545, 1.24376907])
    arm.safe_move_to_position(f)
    arm.safe_move_to_position(mid_f)
    arm.safe_move_to_position(new_dynamic)
    
  

    

    
      

      
    


    # for k in range (4):
    #     jointposition, T0e= fk.forward(start)
    #     g = np.dot(T0e, H_ee)
    #     goal = np.dot(g, H_block)
    #     q= move_arm(goal,start)

    #     state = arm.get_gripper_state()
    #     pos = state['position']
    #     length=abs(pos[0]+pos[1])

    #     if(length>=0.05): #or length<=0.4):
    #         #arm.open_gripper()
        
    #         arm.exec_gripper_cmd(0.049, 15)
        
    #     arm.safe_move_to_position(start)
        

    #     #Change force value as required
    #     #arm.safe_move_to_position(start)
    #     #for i in range(4):
    #     final = np.array([[1, 0, 0,  0.259],
    #                         [0, -1, 0,  0.652],
    #                         [0, 0, -1, 0.350+k*0.05],
    #                         [0, 0,  0,  1]])
    
    #     q=move_arm(final, start)
    #     arm.exec_gripper_cmd(0.15, 15)
    #     arm.safe_move_to_position(start)
    #     k=k+1







if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    #start_position = np.array([-0.87673 ,-1.18955, -0.85865,-2.95039, 0.52488 , 0.83928, -1.12805])
    #start_position = np.array([-0.87673 ,-1.18955, -0.85865,-2.95039, 0.52488 , 0.83928, -1.12805])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!
    
    
    #map_struct = loadmap("maps/map1.txt")
    # STUDENT CODE HERE
    #start = np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
    
    
    if team == 'red':
      #start = np.array([-3.11176606e-01 ,-3.57905471e-02 ,-7.56097364e-03 ,-1.90897320e+00, -2.68731754e-04 , 1.86965151e+00 , 4.66735091e-01])
      #start = np.array([-0.87673 ,-1.18955, -0.85865,-2.95039, 0.52488 , 0.83928, -1.12805])
      start = np.array([-0.16816,  0.24084, -0.16146, -1.04058,  0.04005,  1.27863,  0.4719])
      arm.safe_move_to_position(start)
      
    
      # get the transform from camera to panda_end_effector
      H_ee_camera = detector.get_H_ee_camera()
      
      # #print(H_ee_camera)
      i=0
      k=0
      arm.exec_gripper_cmd(0.15, 15)
      #print(detector.get_detections())
      # Detect some blocks...
      A1 = detector.get_detections()
      A2 = detector.get_detections()
      A3 = detector.get_detections()
      xyz1 = np.zeros((4,3))
      xyz2 = np.zeros((4,3))
      xyz3 = np.zeros((4,3))
      mean_xyz = np.zeros((4,3))
      #print(np.shape(A1))
      
      for k in range(np.shape(A1)[0]):
        xyz1[k,:] = A1[k][1][0:3,3]
        xyz2[k,:] = A2[k][1][0:3,3]
        xyz3[k,:] = A3[k][1][0:3,3]
      #print(A1[3][1][0:3,3])
      #print(xyz2)
      mean_xyz = (xyz1 + xyz2 + xyz3)/3
      for k in range(np.shape(A1)[0]):
        A2[k][1][0:3,3] = mean_xyz[k,:]

      for (name, pose) in A2:
          
          #print(name,'\n',pose)
          St_block = pose
          #print(pose)
          
          st1= pick_and_place(team, St_block, H_ee_camera,i)
          i=i+1
      j=0
      new_dynamic=np.array([ 1.55884611 , 0.96035708 , 0.19518399 ,-0.08630148, -0.18476489 , 1.04486523, 2.36358211])
      arm.safe_move_to_position(new_dynamic)
      for j in range(2):
        st2=pick_and_place_dynamic(team,H_ee_camera, j)
        j=j+1
    

    else:
      start = np.array([ 0.22989307 , 0.24263811 , 0.08206602, -1.03253644, -0.02059894 ,  1.27409853, 1.08894914])
      #start = np.array([-0.87673 ,-1.18955, -0.85865,-2.95039, 0.52488 , 0.83928, -1.12805])
      arm.safe_move_to_position(start)
    
      # get the transform from camera to panda_end_effector
      H_ee_camera = detector.get_H_ee_camera()
      
      #print(H_ee_camera)
      i=0
      k=0
      arm.exec_gripper_cmd(0.15, 15)
      #print(detector.get_detections())
      # Detect some blocks...
      A1 = detector.get_detections()
      A2 = detector.get_detections()
      A3 = detector.get_detections()
      xyz1 = np.zeros((4,3))
      xyz2 = np.zeros((4,3))
      xyz3 = np.zeros((4,3))
      mean_xyz = np.zeros((4,3))
      #print(np.shape(A1))
      
      for k in range(np.shape(A1)[0]):
        xyz1[k,:] = A1[k][1][0:3,3]
        xyz2[k,:] = A2[k][1][0:3,3]
        xyz3[k,:] = A3[k][1][0:3,3]
      #print(A1[3][1][0:3,3])
      #print(xyz2)
      mean_xyz = (xyz1 + xyz2 + xyz3)/3
      for k in range(np.shape(A1)[0]):
        A2[k][1][0:3,3] = mean_xyz[k,:]

      for (name, pose) in A2:
          
          #print(name,'\n',pose)
          St_block = pose
          #print(pose)
          
          st1= pick_and_place(team, St_block, H_ee_camera,i)
          i=i+1
      
      new_dynamic=np.array([0.74938978, -0.88994275, -1.71396112 ,-1.25485352 ,-0.8824145 ,  1.48073024, 1.20047301])
        
      arm.safe_move_to_position(new_dynamic)
      
      for j in range(2):
        st2=pick_and_place_dynamic(team,H_ee_camera, j)
        j=j+1

    # Move around...

    # END STUDENT CODEc