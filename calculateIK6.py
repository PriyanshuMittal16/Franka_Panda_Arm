import numpy as np
import math as m
import random
from math import pi, acos
#from scipy.linalg import null_space
from copy import deepcopy
from scipy.spatial.transform import Rotation

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK


fk = FK()

class IK:
    @staticmethod
    def inverse(current_position,T0et):
        #current_position = [0,0,0,-1.57,0,1.57,0.785]
        #current_position = [-0.03450483,0.22539452,-0.14064529,-1.77050944,0.03436498,1.99356833,0.78106695]
        #target = [1,1,1,0,0,0,2]
        jointPositionscp, T0ecp = fk.forward(current_position)
        #jointpositionst, T0et = fk.forward(target)
        #print(np.shape(T0ecp))
        acp = T0ecp[0:3,3]
        #print(acp)
        at = T0et[0:3,3]
        itr = 0
        #error_dash = at - acp
        #error_dash = error_dash.reshape(3,1)

        Rcp = T0ecp[0:3,0:3]
        Rt = T0et[0:3,0:3]
        Rcpi = np.linalg.inv(Rcp)
        Rprime = np.dot(Rt,Rcpi)
        thetap = (Rprime[0,0] + Rprime[1,1] +Rprime[2,2] - 1) * 0.5
        #print(thetap)
        theta = m.acos(thetap)
        #print(theta)
        s = 1/(2*m.sin(theta))
        s = float(s)
        k = np.array([[Rprime[2,1] - Rprime[1,2]],
                    [Rprime[0,2] - Rprime[2,0]],
                    [Rprime[1,0] - Rprime[0,1]]])
            
        k1 = theta * s * k

        error_dash = T0et - T0ecp

        tolerance = [[0.002, 0.002, 0.002, 0.001],
                    [0.002, 0.002, 0.002, 0.001],
                    [0.002, 0.002, 0.002, 0.001],
                    [0.002, 0.002, 0.002, 0.001]]
        #print(np.shape(error_dash))


        while ((np.abs(error_dash) >= tolerance).any()):
            Jvw = (calcJacobian(current_position))
            Jvwt = np.transpose(Jvw)
            #print(np.shape(Jvwt))
            acp = T0ecp[0:3,3]
            alpha = 0.5
            a3 = at - acp
            a3 = a3.reshape(3,1)
            
            Rcp = T0ecp[0:3,0:3]
            Rt = T0et[0:3,0:3]
            Rcpi = np.linalg.inv(Rcp)
            Rprime = np.dot(Rt,Rcpi)
            #Rprime = Rt - Rcp
            thetap = (Rprime[0,0] + Rprime[1,1] +Rprime[2,2] - 1) * 0.5
            #print(thetap)
            theta = m.acos(thetap)
            #print(theta)
            s = 1/(2*m.sin(theta))
            s = float(s)
            k = np.array([[Rprime[2,1] - Rprime[1,2]],
                        [Rprime[0,2] - Rprime[2,0]],
                        [Rprime[1,0] - Rprime[0,1]]])
            
            k1 = theta * s * k
            #print(np.linalg.norm(k1))
            b1 = np.append(a3,k1,0)
            #print(np.shape(b1))
            

            a4 = np.dot(Jvwt,b1)
            #print(np.shape(a4))
            dq = alpha*a4
            #print(np.shape(dq))
            # current_new = current_position
            current_position = np.reshape(current_position,(7,1)) + dq
            #print(np.shape(current_position))
            # now = np.abs(current_new - current_position)
            #print(now)

            # if (now[0,]>0.0001 or now[1,]>0.0001 or now[2,]>0.0001 or now[3,]>0.0001 or now[4,]>0.0001 or now[5,]>0.0001 or now[6,]>0.0001):
            #     flag = 0
            # else:
            #     flag = 1
            # if (flag == 1):
            #     print("minima reached")
            #     walk = random.randrange(1, 2)
            #     current_position  = current_position  + np.array([0.01,0.35,0.1,0.05,0.01,0.1,0.01])*walk
            jointPositionscp, T0ecp = fk.forward(current_position)
            #acp = T0ecp[0:3,3]
            error_dash = T0et - T0ecp
            
            #error_dash = at - acp
            #error_dash = error_dash.reshape(3,1)
            #print(error_dash)

            if (current_position[0]>2.8973 or current_position[0]<-2.8973):
                current_position[0] = 1
                #print("hi")
            elif (current_position[1]>1.7628 or current_position[1]<-1.7628):
                current_position[1] = 1
                #print("hi")
            elif (current_position[2]>2.8973 or current_position[2]<-2.8973):
                current_position[2] = 1
                #print("hi")
            elif (current_position[3]>-0.0698 or current_position[3]<-3.0718):
                current_position[3] = -1
                #print("hi")
            elif (current_position[4]>2.8973 or current_position[4]<-2.8973):
                current_position[4] = 1
                #print("hi")
            elif (current_position[5]>3.7525 or current_position[5]<-0.0175):
                current_position[5] = 1
                #print("hi")
            elif (current_position[6]>2.8973 or current_position[6]<-2.8973):
                current_position[6] = 1
                #print("hi")

            itr = itr + 1
            if (itr > 40000):
                break
        jointPositionscp, T0ecp = fk.forward(current_position)
        #acp = T0ecp[0:3,3]
        # #print(acp)
        #at = T0et[0:3,3]
        # itr = 0
        error_dash = T0et - T0ecp
        #error_dash = at - acp
        print(itr)
        print(error_dash)
        #print(current_position)





        return current_position

if __name__ == '__main__':
    q = np.array([1, 1, 0, -1, 0, 0, 3.224])
    T0et = np.array([[ 0,  1,  0,  0],
                    [ 1, 0, 0,  0.305],
                    [ 0,  0, -1,  0.4],
                    [ 0,  0,  0,  1]])
    #q = np.array([0, 0, 0, 0, 0, 0, np.pi/4])
    
    
    
    k = IK.inverse(q,T0et)
    print(k.reshape(1,7))

