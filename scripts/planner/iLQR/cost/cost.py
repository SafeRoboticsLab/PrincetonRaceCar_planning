import jax
from jax import numpy as jnp
import numpy as np
from jaxlib.xla_extension import DeviceArray
from typing import Union

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
			self, trajectory: Union[np.ndarray, DeviceArray], 
            controls: Union[np.ndarray, DeviceArray], 
            path_refs: Union[np.ndarray, DeviceArray], 
            obs_refs: list = None
	) -> float:
        '''
		Given a state, control, and references, return the sum of the cost.
		Input:
			trajectory: (dim_x, N) array of state trajectory
			controls: (dim_u, N) array of control sequence
			path_refs: (dim_ref, N) array of references (e.g. reference path, reference velocity, etc.)
			obs_refs: -Optional- (num_obstacle, 2, N) List of obstacles
		return:
			cost: float, sum of the running cost over the trajectory
		'''
        
        state_cost = self.state_cost.get_traj_cost(trajectory, controls, path_refs)
        
        control_cost = self.control_cost.get_traj_cost(trajectory, controls, path_refs)
        
        obstacle_cost = self.obstacle_cost.get_traj_cost(trajectory, controls, obs_refs)
        
        return state_cost + control_cost + obstacle_cost
    
    def get_derivatives_jax(
			self, trajectory: Union[np.ndarray, DeviceArray], 
            controls: Union[np.ndarray, DeviceArray], 
            path_refs: Union[np.ndarray, DeviceArray], 
            obs_refs: list = None
	) -> tuple:
        '''
		Given a state, control, and references, return Jacobians and Hessians of cost function
		Input:
			trajectory: (dim_x, N) array of state trajectory
			controls: (dim_u, N) array of control sequence
			path_refs: (dim_ref, N) array of references (e.g. reference path, reference velocity, etc.)
			obs_refs: -Optional- (num_obstacle, 2, N) List of obstacles
		return:
			q: DeviceArray, jacobian of cost function w.r.t. trajectory
            r: DeviceArray, jacobian of cost function w.r.t. controls
            Q: DeviceArray, hessian of cost function w.r.t. trajectory
            R: DeviceArray, hessian of cost function w.r.t. controls
            H: DeviceArray, hessian of cost function w.r.t. trajectory and controls
		'''
        
        (state_q, state_r, state_Q, 
            state_R, state_H) = self.state_cost.get_derivatives_jax(trajectory, controls, path_refs)
        
        (ctrl_q, ctrl_r, ctrl_Q,
            ctrl_R, ctrl_H) = self.control_cost.get_derivatives_jax(trajectory, controls, path_refs)
        
        (obs_q, obs_r, obs_Q, 
            obs_R, obs_H) = self.obstacle_cost.get_derivatives_jax(trajectory, controls, obs_refs)
        
        return (state_q + ctrl_q + obs_q,
                state_r + ctrl_r + obs_r,
                state_Q + ctrl_Q + obs_Q,
                state_R + ctrl_R + obs_R,
                state_H + ctrl_H + obs_H)
        
    def get_derivatives_np(
			self, trajectory: np.ndarray, controls: np.ndarray, path_refs: np.ndarray, obs_refs: list = None
	) -> tuple:
        '''
		Given a state, control, and references, return Jacobians and Hessians of cost function
		Input:
			trajectory: (dim_x, N) array of state trajectory
			controls: (dim_u, N) array of control sequence
			path_refs: (dim_ref, N) array of references (e.g. reference path, reference velocity, etc.)
			obs_refs: -Optional- (num_obstacle, 2, N) List of obstacles
		return:
			q: np.ndarray, jacobian of cost function w.r.t. trajectory
            r: np.ndarray, jacobian of cost function w.r.t. controls
            Q: np.ndarray, hessian of cost function w.r.t. trajectory
            R: np.ndarray, hessian of cost function w.r.t. controls
            H: np.ndarray, hessian of cost function w.r.t. trajectory and controls
		'''
        q, r, Q, R, H = self.get_derivatives_jax(trajectory, controls, path_refs, obs_refs)
        return np.asarray(q), np.asarray(r), np.asarray(Q), np.asarray(R), np.asarray(H)