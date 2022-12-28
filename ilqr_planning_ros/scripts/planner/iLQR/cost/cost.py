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

class Cost(BaseCost):
    def __init__(self, config):
        super().__init__()
        
        self.state_cost = StateCost(config)
        self.control_cost = ControlCost(config)
        
    def update_obstacles(self, obstacles):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def get_stage_cost(self, state, ctrl, ref, time_idx):
        
        state_cost = self.state_cost.get_stage_cost(state, ctrl, ref, time_idx)
        control_cost = self.control_cost.get_stage_cost(state, ctrl, ref, time_idx)
        
        return state_cost + control_cost
        