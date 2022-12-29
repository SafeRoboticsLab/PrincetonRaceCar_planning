import numpy as np
import rospy

'''
    A container class to store the feedback policy
'''
class Policy():
    def __init__(self, x, u, K, t0, dt, N) -> None:
        self.nominal_x = x
        self.nominal_u = u
        self.K = K
        self.t0 = t0
        self.dt = dt
        self.N = N
    
    def get_policy(self, t):
        '''
        Return the policy at time t
        '''
        i = int(np.floor((t-self.t0).to_sec()/self.dt))
        if i>= self.N:
            rospy.logwarn("Try to retrive policy beyond horizon")
            x_i = self.nominal_x[:,-1]
            x_i[2:] = 0 # set velocity to zero
            u_i = np.zeros_like(self.nominal_u[:,0])
            K_i = np.zeros_like(self.K[:,:,0])
        else:
            x_i = self.nominal_x[:,i]
            u_i = self.nominal_u[:,i]
            K_i = self.K[:,:,i]

        return x_i, u_i, K_i
