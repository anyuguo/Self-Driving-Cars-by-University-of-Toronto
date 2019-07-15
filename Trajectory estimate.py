# -*- coding: utf-8 -*-
"""
Created on Mon Jul 5 13:22:56 2019

@author: AskLan
"""


import pickle
import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#def wraptopi(x):
#    if x > np.pi:
#        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
#    elif x < -np.pi:
#        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
#    return x

def wraptopi(x):
    if x > np.pi or x<-np.pi:
#        x = x - 2 * np.pi * np.floor(x/(2*np.pi) + 0.5)
        x = ((x / np.pi + 1) % 2 - 1) * np.pi
    return x

#def wraptopi(x):
#    return max(-np.pi, min(x, np.pi))

def measurement_update(lk, rk, bk, P_check, x_check):
    
    xl = lk[0]
    yl = lk[1]
    
    x = x_check[0,0]
    y = x_check[1,0]
    theta = x_check[2,0]
    
    d = 0
    
    # 1. Compute measurement Jacobian
    p1 = xl - x - d * math.cos(theta)
    p2 = yl - y - d * math.sin(theta)
    den = p1**2 + p2**2
    den_sq = math.sqrt(den)

    H_k=np.mat([[-p1/den_sq, -p2/den_sq, (p1 * d * math.sin(theta) - p2 * d * math.cos(theta))/den_sq],
               [p2/den, -p1/den, -1-d*(math.sin(theta) * p2 + math.cos(theta) * p1)/den]])
    M_k=np.identity(2)

    # 2. Compute Kalman Gain
    K_k = P_check @ H_k.T @ inv(H_k @ P_check @ H_k.T + M_k @ cov_y @ M_k.T)

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    y_measure=np.mat([[rk],
                      [bk]])
    
    yk=np.mat([[den_sq],
              [wraptopi(np.arctan2(p2,p1))-theta]])
    
    x_check = x_check + K_k @ (y_measure-yk)
    
    x_check[2,0] = wraptopi(x_check[2,0])
        
    # 4. Correct covariance
    P_check = (np.identity(3) - K_k @ H_k) @ P_check

    return x_check, P_check

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.01  # translation velocity variance  
om_var = 0.01  # rotational velocity variance 
r_var = 0.008  # range measurements variance
b_var = 10000  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


#### 5. Main Filter Loop #######################################################################
x_check = x_est[0,:]
x_check = np.mat([[x_check[0]],
                 [x_check[1]],
                 [x_check[2]]])
P_check=P_est[0]

for k in range(1, len(t)):  
    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    theta = x_check[2,0]
    f = np.mat([[math.cos(theta), 0], 
                [math.sin(theta), 0],
                [0, 1]])
    
    u = np.mat([[v[k-1]],
                [om[k-1]]])
    
    x_check = x_check + f * u * delta_t
    x_check[2,0] = wraptopi(x_check[2,0])

    # 2. Motion model jacobian with respect to last state
    F_km = np.mat([[1, 0, -delta_t * math.sin(theta) * v[k-1]],
                   [0, 1, delta_t * math.cos(theta) * v[k-1]],
                   [0, 0, 1]])

    # 3. Motion model jacobian with respect to noise
    L_km = np.mat([[delta_t * math.cos(theta), 0],
                   [delta_t * math.sin(theta), 0],
                   [0, delta_t]])

    # 4. Propagate uncertainty
    P_check = F_km @ P_check @ F_km.T + L_km @ Q_km @ L_km.T

    # 5. Update state estimate using available landmark measurements

    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)
        
    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0,0]
    x_est[k, 1] = x_check[1,0]
    x_est[k, 2] = x_check[2,0]
    P_est[k, :, :] = P_check

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()