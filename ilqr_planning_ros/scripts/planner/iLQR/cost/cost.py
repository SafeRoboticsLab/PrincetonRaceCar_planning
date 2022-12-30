from abc import ABC, abstractmethod
from functools import partial
from typing import Optional
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
import numpy as np

from .base_cost import BaseCost
from .state_cost import StateCost
from .control_cost import ControlCost
from .obstacle_cost import ObstacleCost

import time

class Cost(BaseCost):
    def __init__(self, config):
        super().__init__()
        
        self.state_cost = StateCost(config)
        self.control_cost = ControlCost(config)
        
        self.obs_cost = ObstacleCost(config)
        
        # Progress Cost
        self.dim_progress = config.dim_progress
        self.progress_weight = config.progress_weight/(config.n * config.dt)
        
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
        
    def update_obstacles(self, obstacles):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def get_running_cost(self, state, ctrl, ref):
        
        state_cost = self.state_cost.get_running_cost(state, ctrl, ref)
        control_cost = self.control_cost.get_running_cost(state, ctrl, ref)
        obs_cost = self.obs_cost.get_running_cost(state, ctrl, ref)
        
        return state_cost + control_cost + obs_cost
        