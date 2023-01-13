import jax
from jax import numpy as jnp
import numpy as np
from jaxlib.xla_extension import DeviceArray

from .state_cost import StateCost
from .control_cost import ControlCost
from .obstacle_cost import ObstacleCost

import time

class Cost():
    def __init__(self, config):
        super().__init__()
        
        self.state_cost = StateCost(config)
        self.control_cost = ControlCost(config)
        self.obstacle_cost = ObstacleCost(config)
    
    def get_traj_cost(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, obs_refs: DeviceArray = None
	) -> float:
        '''
		Given a state, control, and time index, return the sum of the cost.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			cost: float
		'''
        t0 = time.time()
        state_cost = self.state_cost.get_traj_cost(states, ctrls, refs)
        t1 = time.time()
        control_cost = self.control_cost.get_traj_cost(states, ctrls, refs)
        t2 = time.time()
        obstacle_cost = self.obstacle_cost.get_traj_cost(states, ctrls, obs_refs)
        # print("get cost: ", t1 - t0, t2-t1, time.time()-t2)

        return state_cost + control_cost + obstacle_cost

    def get_derivatives(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, obs_refs: DeviceArray = None
	) -> DeviceArray:
        # Get Jacobians and Hessians of each cost function then sum them up
        
        (state_cx, state_cu, state_cxx, 
            state_cuu, state_cux) = self.state_cost.get_derivatives(states, ctrls, refs)
        
        (ctrl_cx, ctrl_cu, ctrl_cxx,
            ctrl_cuu, ctrl_cux) = self.control_cost.get_derivatives(states, ctrls, refs)
        
        (obs_cx, obs_cu, obs_cxx, 
            obs_cuu, obs_cux) = self.obstacle_cost.get_derivatives(states, ctrls, obs_refs)
        
        return (state_cx + ctrl_cx + obs_cx,
                state_cu + ctrl_cu + obs_cu,
                state_cxx + ctrl_cxx + obs_cxx,
                state_cuu + ctrl_cuu + obs_cuu,
                state_cux + ctrl_cux + obs_cux)
