import pickle
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def wraptopi(x):
    return ((x / np.pi + 1) % 2 - 1) * np.pi

def measurement_update(lk, rk, bk, P_check, x_check):
    bk = wraptopi(bk)
    xl = lk[0]
    yl = lk[1]
    
    x_check[2,0] = wraptopi(x_check[2,0])
    
    xk = x_check[0,0]
    yk = x_check[1,0]
    theta = x_check[2,0]
    
    d = 0
    
    # 1. Compute measurement Jacobian
    R_k = cov_y
    
    M_k = np.eye(2)
    
    p1 = xl - xk - d * np.cos(theta)
    p2 = yl - yk - d * np.sin(theta)
    den = p1**2 + p2**2
    den_sq = np.sqrt(den)
    
    H_k = np.zeros([2,3])
    H_k[0,0] = -p1/den_sq
    H_k[0,1] = -p2/den_sq
    H_k[0,2] = d*(p1*np.sin(theta) - p2*np.cos(theta))/den_sq
    H_k[1,0] = p2/den
    H_k[1,1] = -p1/den
    H_k[1,2] = -1 - d * (np.sin(theta)*p2 + np.cos(theta)*p1)/den
    
    # 2. Compute Kalman Gain
    K_k = P_check @ H_k.T @ inv(H_k @ P_check @ H_k.T + M_k @ R_k @ M_k.T)
    
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    yk = np.array([rk, bk]).reshape(2,1)
    
    h = np.zeros([2,1])
    h[0,0] = den_sq
    h[1,0] = wraptopi(np.arctan2(p2, p1) - theta)
    
    x_hat = np.zeros([3,1])
    x_hat = x_check + K_k @ (yk - h)
    x_hat[2,0] = wraptopi(x_hat[2,0])
    # 4. Correct covariance
    P_hat = np.zeros([3,3])
    P_hat = (np.eye(3) - K_k @ H_k) @ P_check
    
    
    return x_hat, P_hat
    

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init = data['x_init']  # initial x position [m]
y_init = data['y_init']  # initial y position [m]
th_init = data['th_init']  # initial theta position [rad]

# input signal
v = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]


v_var = 0.01  # translation velocity variance
om_var = 0.01  # rotational velocity variance
r_var = 0.1  # range measurements variance
b_var = 3e-6  # bearing measurement variance

Q_km = np.diag([v_var, om_var])  # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init])  # initial state
P_est[0] = np.diag([1, 1, 0.1])  # initial state covariance


#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):
    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    
    T = delta_t
    
    x_check = x_est[k-1].reshape(3, 1)
    theta_minus = x_check[2,0]
    P_check = P_est[k-1]
    
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    f = np.zeros([3,2])
    f[0,0] = T * np.cos(theta_minus)
    f[1,0] = T * np.sin(theta_minus)
    f[2,1] = T
    
    u = np.array([v[k-1], om[k-1]]).reshape(2,1)
    
    x_check = x_check + f @ u
    x_check[2,0] = wraptopi(x_check[2,0])
    
    # 2. Motion model jacobian with respect to last state
    F_km = np.eye(3)
    F_km[0,2] = - T * v[k-1] * np.sin(theta_minus)
    F_km[1,2] = T * v[k-1] * np.cos(theta_minus)
    
    # 3. Motion model jacobian with respect to noise
    L_km = f
    
    # 4. Propagate uncertainty
    P_check = F_km @ P_check @ F_km.T + L_km @ Q_km @ L_km.T
    
    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)
    
    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0,0]
    x_est[k, 1] = x_check[1,0]
    x_est[k, 2] = wraptopi(x_check[2,0])
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

with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)