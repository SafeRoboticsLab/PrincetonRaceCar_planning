from abc import ABC, abstractmethod
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
import numpy as np

from .base_cost import BaseCost
from .base_cost import exp_linear_cost
class SingleObstacleCost(BaseCost):
    def __init__(self, config):
        '''
        Obtain the cost and its gradient w.r.t to a single obstacle given a state, control,
        '''
        super().__init__()
        
                                                
        self.cost_func = exp_linear_cost
        
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
            ref: (5,) obstacle reference 
            time_idx: int (1)
        return:
            cost: float
        '''
        psi = state[3]
        rot_mat = jnp.array([[jnp.cos(psi), -jnp.sin(psi)],
                            [jnp.sin(psi), jnp.cos(psi)]])
        
        ego_pt = ref[:2]
        obs_pt = ref[2:4]
        dis_ref = ref[4]
        
        ego_pt_global = jnp.matmul(rot_mat, ego_pt) + state[:2]
        
        diff = obs_pt - ego_pt_global

        # handle the case when distance is 0 (when the obstacle touch each other)
        # THis leads to nan in the gradient
        dis = jnp.linalg.norm(diff)
        y = jnp.where(dis >= 1e-4, dis, 1e-4)

        y= -1*jnp.sign(dis_ref)*y   

        # Do not calculate gradient w.r.t velocity
        v_no_grad = jax.lax.stop_gradient(state[2])
        obs_b = self.obs_b*jnp.maximum(v_no_grad, 1)
        
        return exp_linear_cost(y, self.obs_a, obs_b)
    
class ObstacleCost():
    def __init__(self, config):
        self.single_obstacle_cost = SingleObstacleCost(config)
        
    def get_traj_cost(
			self, states: DeviceArray, ctrls: DeviceArray, obs_refs: DeviceArray
	) -> float:
        '''
		Given a state, control, and time index, return the sum of the cost.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (num_obstacle, dim_ref, N) List of references
		return:
			cost: float
		'''
        cost = 0
        if obs_refs is not None:
            num_obstacle = obs_refs.shape[0]
            for i in range(num_obstacle):
                cost += self.single_obstacle_cost.get_traj_cost(states, ctrls, obs_refs[i])
        
        return cost

    def get_derivatives(
            self, states: DeviceArray, ctrls: DeviceArray, obs_refs: DeviceArray
    ) -> DeviceArray:
        
        cx = 0
        cu = 0
        cxx = 0
        cuu = 0
        cux = 0
        
        if obs_refs is not None:
            num_obstacle = obs_refs.shape[0]
            
            for i in range(num_obstacle):
                cx_, cu_, cxx_, cuu_, cux_ = self.single_obstacle_cost.get_derivatives(states, ctrls, obs_refs[i])
                cx += cx_
                cu += cu_
                cxx += cxx_
                cuu += cuu_
                cux += cux_
        return (cx, cu, cxx, cuu, cux)