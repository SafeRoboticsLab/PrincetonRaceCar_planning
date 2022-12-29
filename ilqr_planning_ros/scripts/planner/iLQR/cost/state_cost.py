from abc import ABC, abstractmethod
from functools import partial
from typing import Optional
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
import numpy as np

from .base_cost import BaseCost
from .base_cost import quadratic_cost, huber_cost

def raise_error(cost_type, cost_name):
    raise NotImplementedError(
        f'Cost type {cost_type} not implemented for {cost_name} of STATE COST.'
    )

class StateCost(BaseCost):
    def __init__(self, config):
        super().__init__()
        
        # Reference Path Cost
        self.dim_path_slope = config.dim_path_slope
        self.dim_closest_pt_x = config.dim_closest_pt_x
        self.dim_closest_pt_y = config.dim_closest_pt_y
        self.path_weight = config.path_weight
        self.path_delta = config.path_huber_delta
        if config.path_cost_type == 'quadratic':
            self.path_cost_func = quadratic_cost
        elif config.path_cost_type == 'huber':
            self.path_cost_func = huber_cost
        else:
            raise_error(config.path_cost_type, 'PATH_COST')
            
        # Reference Velocity Cost
        self.dim_vel_ref = config.dim_vel_ref
        self.vel_weight = config.vel_weight
        self.vel_delta = config.vel_huber_delta
        if config.vel_cost_type == 'quadratic':
            self.vel_cost_func = quadratic_cost
        elif config.vel_cost_type == 'huber':
            self.vel_cost_func = huber_cost
        else:
            raise_error(config.vel_cost_type, 'VEL_COST')
        
        
        self.weight = jnp.array(config.ctrl_cost_weight) # shape of (dim_u)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_stage_cost(self, state, ctrl, ref, time_idx):
        '''
        Given a state, control, and time index, return the cost.
        Input:
            state: (dim_x) - [x, y, v, psi, delta]
            ctrl: (dim_u)
            ref: (dim_ref) reference 
            time_idx: int (1)
        return:
            cost: float
        '''
        # Cost for the vehicle's deviation from the reference path
        slope = ref[self.dim_path_slope]
        closest_pt_x = ref[self.dim_closest_pt_x]
        closest_pt_y = ref[self.dim_closest_pt_y]
        sr = jnp.sin(slope)
        cr = jnp.cos(slope)
        path_dev = sr * (state[0] - closest_pt_x) - cr *(state[1] - closest_pt_y)
        path_cost = self.path_cost_func(path_dev, self.path_weight, self.path_delta)
        
        # Cost for the vehicle's deviation from the reference velocity
        vel_ref = ref[self.dim_vel_ref]
        vel_dev = state[2] - vel_ref
        vel_cost = self.vel_cost_func(vel_dev, self.vel_weight, self.vel_delta)
        
        # jax.debug.print("step {i}, vel: {v}, vel_ref: {r}, vel_cost: {y}",
        #                 i = time_idx, v = state[2],  r = vel_ref, y = vel_cost)
        
        
        return path_cost + vel_cost
    
    