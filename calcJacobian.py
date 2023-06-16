import numpy as np
import math as m
#from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))
    t1=q_in[0]
    t2=q_in[1]
    t3=q_in[2]
    t4=q_in[3]
    t5=q_in[4]
    t6=q_in[5]
    t7=q_in[6]

    #Robot Geometry 
    d1=0.333
    d3=0.316
    d5=0.384
    d7=0.210
    a3=0.0825
    a4=a3
    a6=0.088
    
    #Homogeneous Transformations, DH parameters are taken with respect to DH table in LAB 2
    A1 = np.array([m.cos(t1),0,-m.sin(t1),0,m.sin(t1),0,m.cos(t1),0,0,-1,0,d1,0,0,0,1]).reshape(4,4) 
    A2 = np.array([m.cos(t2),0,m.sin(t2),0,m.sin(t2),0,-m.cos(t2),0,0,1,0,0,0,0,0,1]).reshape(4,4)
    A3 = np.array([m.cos(t3),0,m.sin(t3),a3*m.cos(t3),m.sin(t3),0,-m.cos(t3),a3*m.sin(t3),0,1,0,d3,0,0,0,1]).reshape(4,4) 
    A4 = np.array([m.cos(t4),0,-m.sin(t4),-a4*m.cos(t4),m.sin(t4),0,m.cos(t4),-a4*m.sin(t4),0,-1,0,0,0,0,0,1]).reshape(4,4) 
    A5 = np.array([m.cos(t5),0,m.sin(t5),0,m.sin(t5),0,-m.cos(t5),0,0,1,0,d5,0,0,0,1]).reshape(4,4) 
    A6 = np.array([m.cos(t6),0,m.sin(t6),a6*m.cos(t6),m.sin(t6),0,-m.cos(t6),a6*m.sin(t6),0,1,0,0,0,0,0,1]).reshape(4,4) 
    A7 = np.array([m.cos(t7),-m.sin(t7),0,0,m.sin(t7),m.cos(t7),0,0,0,0,1,d7,0,0,0,1]).reshape(4,4)
    

    #End Effector Homogeneous Transformation
    T70=np.identity(4)

    T701 = np.matmul(T70,A1)
    T702 = np.matmul(T701,A2)
    T703 = np.matmul(T702,A3)
    T704 = np.matmul(T703,A4)
    T705 = np.matmul(T704,A5)
    T706 = np.matmul(T705,A6)
    T707 = np.matmul(T706,A7)

    #FK complete

    #Extracting the position vector from the homogeneous transformation matrices
    o7l = T707[0:3,3]
    o7 = o7l.reshape(3,1)
    o1l = T701[0:3,3]
    o1 = o1l.reshape(3,1)
    o2l = T702[0:3,3]
    o2 = o2l.reshape(3,1)
    o3l = T703[0:3,3]
    o3 = o3l.reshape(3,1)
    o4l = T704[0:3,3]
    o4 = o4l.reshape(3,1)
    o5l = T705[0:3,3]
    o5 = o5l.reshape(3,1)
    o6l = T706[0:3,3]
    o6 = o6l.reshape(3,1)
    o0 = np.array([0,0,0]).reshape(3,1)

    o01 = o7-o0
    o02 = o7-o1
    o03 = o7-o2
    o04 = o7-o3
    o05 = o7-o4
    o06 = o7-o5
    o07 = o7-o6

    #Extracting the rotation matrix from the homogeneous transformation matrices
    z6l = T706[0:3,2]
    z6 = z6l.reshape(3,1)
    z5l = T705[0:3,2]
    z5 = z5l.reshape(3,1)
    z4l = T704[0:3,2]
    z4 = z4l.reshape(3,1)
    z3l = T703[0:3,2]
    z3 = z3l.reshape(3,1)
    z2l = T702[0:3,2]
    z2 = z2l.reshape(3,1)
    z1l = T701[0:3,2]
    z1 = z1l.reshape(3,1)
    z7l = T707[0:3,2]
    z7 = z7l.reshape(3,1)
    z0 = np.array(([0,0,1])).reshape((3,1))
    #z0 = np.transpose(z0)

    Jv1 = np.cross(z0.T,o01.T).T
    Jv2 = np.cross(z1.T,o02.T).T
    Jv3 = np.cross(z2.T,o03.T).T
    Jv4 = np.cross(z3.T,o04.T).T
    Jv5 = np.cross(z4.T,o05.T).T
    Jv6 = np.cross(z5.T,o06.T).T
    Jv7 = np.cross(z6.T,o07.T).T

    Jw1 = z0
    Jw2 = z1
    Jw3 = z2
    Jw4 = z3
    Jw5 = z4
    Jw6 = z5
    Jw7 = z6
    
    Jva = np.append(Jv1,Jv2,1)
    Jvb = np.append(Jva,Jv3,1)
    Jvc = np.append(Jvb,Jv4,1)
    Jvd = np.append(Jvc,Jv5,1)
    Jve = np.append(Jvd,Jv6,1)
    Jvf = np.append(Jve,Jv7,1)

    Jwa = np.append(Jw1,Jw2,1)
    Jwb = np.append(Jwa,Jw3,1)
    Jwc = np.append(Jwb,Jw4,1)
    Jwd = np.append(Jwc,Jw5,1)
    Jwe = np.append(Jwd,Jw6,1)
    Jwf = np.append(Jwe,Jw7,1)

    J = np.append(Jvf,Jwf,0)

    #Jw = np.array([Jw1,Jw2,Jw3,Jw4,Jw5,Jw6,Jw7]).reshape(3,7)

    #print(J)
    ## STUDENT CODE GOES HERE

    return J

if __name__ == '__main__':
    q = np.array([0, 0, 0, 0, 0, 0, 0])
    #q = np.array([0, 0, 0, 0, 0, 0, np.pi/4])
    
    
    
    k = calcJacobian(q)
    #print(np.round(k,3))
