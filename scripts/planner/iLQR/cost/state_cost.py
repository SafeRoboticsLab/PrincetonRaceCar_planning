from abc import ABC, abstractmethod
from functools import partial
from typing import Optional
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
import numpy as np

from .base_cost import BaseCost
from .base_cost import quadratic_cost, huber_cost, exp_linear_cost

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
        
        # Lateral Acceleration Cost
        self.wheelbase = config.wheelbase
        self.lat_accel_thres = config.lat_accel_thres #float
        self.lat_accel_a = config.lat_accel_a # float
        self.lat_accel_b = config.lat_accel_b # float
        self.lat_accel_cost_func = exp_linear_cost
        
        # Progress Cost
        self.dim_progress = config.dim_progress
        self.progress_weight = config.progress_weight/(config.n * config.dt)

        # Boundary Cost
        self.width = config.width
        self.dim_right_boundary = config.dim_right_boundary
        self.dim_left_boundary = config.dim_left_boundary
        self.lane_boundary_a = config.lane_boundary_a
        self.lane_boundary_b = config.lane_boundary_b
    
    @partial(jax.jit, static_argnums=(0,))
    def get_terminal_cost(
			self, ref: DeviceArray
	) -> float:
        '''
        Since the progress is calulate from PySpline,
        it is intracable to make it work with JAX.
        However, we can locally approximate the progress's derivative
        with method described in the MPCC.
        In this case, we can add a terminal cost to reward the total progress 
        the vehicle has made.
        '''
        progress = ref[self.dim_progress,:]
        return -self.progress_weight * (progress[-1] - progress[0])
    

    @partial(jax.jit, static_argnums=(0,))
    def get_running_cost(self, state, ctrl, ref):
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

        # boundary_cost
        right_boundary = ref[self.dim_right_boundary]
        left_boundary = ref[self.dim_left_boundary]
        b_right = path_dev - right_boundary + self.width/2.0
        b_left = -path_dev - left_boundary + self.width/2.0

        boundary_cost = exp_linear_cost(b_right, self.lane_boundary_a, self.lane_boundary_b) + \
                        exp_linear_cost(b_left, self.lane_boundary_a, self.lane_boundary_b)
        
        # Progress cost
        # This is always zero due to the way the closest point is calculated
        # However, it will help us to calculate the gradient of the cost function
        progress_cost = -self.progress_weight*(cr*(state[0] - closest_pt_x) + sr*(state[1] - closest_pt_y))
        
        # Cost for the vehicle's deviation from the reference velocity
        vel_ref = ref[self.dim_vel_ref]
        vel_dev = state[2] - vel_ref
        vel_cost = self.vel_cost_func(vel_dev, self.vel_weight, self.vel_delta)
        
        # Cost for the vehicle's lateral acceleration
        lat_accel = jnp.abs(state[2]**2 / self.wheelbase * jnp.tan(state[4])) \
                        - self.lat_accel_thres
        
        lat_accel_cost = self.lat_accel_cost_func(lat_accel,
                                    self.lat_accel_a,
                                    self.lat_accel_b)
        # jax.debug.print('{a}, {b}, {c}, {d}',a=path_cost, b=vel_cost, c=lat_accel_cost, d=progress_cost)

        
        return path_cost + vel_cost + lat_accel_cost + progress_cost + boundary_cost