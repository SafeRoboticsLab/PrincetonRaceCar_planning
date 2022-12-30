from abc import ABC, abstractmethod
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
import numpy as np

from .base_cost import BaseCost
from .base_cost import exp_linear_cost


class ObstacleCost(BaseCost):
    def __init__(self, config):
        super().__init__()
        
        # We hard code the ego agent as two circles centered at the front and rear axis of the vehicle
        self.ego_circles_center = jnp.array([[0, config.wheelbase],
                                                [0, 0]])
        
        self.ego_radius = config.radius

        self.cost_func = exp_linear_cost
        
        self.dim_obs_x = config.dim_obs_x
        self.dim_obs_y = config.dim_obs_y
        self.dim_obs_radius = config.dim_obs_radius
        
        self.obs_a = config.obs_a
        self.obs_b = config.obs_b

    @partial(jax.jit, static_argnums=(0,))
    def get_running_cost(
			self, state: DeviceArray, ctrl: DeviceArray, ref: DeviceArray
	) -> float:
        '''
        Given a state, control, and time index, return the cost.
        Input:
            state: (dim_x) state [x, y, v, psi, delta]
            ctrl: (dim_u) control
            ref: (dim_ref) reference 
            time_idx: int (1)
        return:
            cost: float
        '''
        psi = state[3]
        rot_mat = jnp.array([[jnp.cos(psi), -jnp.sin(psi)],
                            [jnp.sin(psi), jnp.cos(psi)]])
        
        circles_center = jnp.matmul(rot_mat, self.ego_circles_center) + state[:2].reshape(2, 1) #[2, 2]
        
        obs_center = jnp.array([[ref[self.dim_obs_x]], [ref[self.dim_obs_y]]])
        
        dis = (self.ego_radius + ref[self.dim_obs_radius]) - jnp.linalg.norm(circles_center - obs_center, axis=0)
        
        # jax.debug.print("obs_cost: {x}", x=dis)
        return jnp.sum(exp_linear_cost(dis, self.obs_a, self.obs_b))