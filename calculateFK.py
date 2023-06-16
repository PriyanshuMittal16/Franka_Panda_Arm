from operator import matmul
import numpy as np
from math import pi
import math

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        
        pass

    def forward(self, q):
        fk = FK()
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        X = np.identity(4);
        Y = np.identity(4);
        A = np.identity(4);
        K = np.zeros((8,3));
        t = [0, (q[0]-pi), -q[1], q[2], (q[3]-(pi/2)+pi/2), -q[4], (q[5]-(pi/2)-pi/2), q[6]-pi/4];
        for i in range(8):
            X = Y;
            a,al,d = fk.dhparams(i);
            A = fk.matrixA(t[i], a, al,d);
            Y = np.matmul(X,A);

            for j in range(3):
                    K[i][j]= Y[j][3];

            if i==2 or i==4 or i==5 or i==6:
                x = [0.195, 0.125, 0.015, 0.051];
                L = Y;
                m = 0;
                if i==2:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,x[0]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];
                
                if i==4:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,-x[1]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];
                
                if i==5:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,-x[2]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];
                
                if i==6:
                    S = [[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,x[3]],
                        [0,0,0,1]];
                    L = np.matmul(L,S)
                    for m in range(3):
                        K[i][m] = L[m][3];

        jointPositions = K;
        T0e = Y;
        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1
    def dhparams(self, i):
        a = [0,0,0,-0.0825,0.0825,0,0.088,0];
        al = [0,-pi/2,pi/2,-pi/2,-pi/2,pi/2,pi/2,0];
        d = [0.141,0.192,0,0.316,0,-0.384,0,0.21];
        return a[i], al[i], d[i];

    def matrixA(self, t,a,al,d):
        cq = math.cos(t);
        cal = math.cos(al);
        sq = math.sin(t);
        sal = math.sin(al);
        A = [[cq, -sq*cal, sq*sal, a*cq], [sq, cq*cal, -cq*sal, a*sq], [0, sal, cal, d], [0, 0, 0, 1]];
        return A

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    #q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #q = np.array([0,0,0,0,0,0,0])
    #q = np.array([0,-1,0,-2,0,1.57,0])
    q = np.array([0.91643413, -1.53079082 ,-1.3649686  ,-1.10494128, -1.62806907 , 1.36921594, -0.33646106])
    joint_positions, T0e = fk.forward(q)

    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)

