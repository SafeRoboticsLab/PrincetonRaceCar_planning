from abc import ABC, abstractmethod
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
import numpy as np

from .base_cost import BaseCost
from .base_cost import quadratic_cost, huber_cost


class LateralAccelCost(BaseCost):
    def __init__(self, config):
        super().__init__()
        self.weight = config.lat_accel_weight #float
        self.delta = config.lat_accel_huber_delta # float
        if config.lat_accel_type == 'quadratic':
            self.cost_func = quadratic_cost
        elif config.lat_accel_type == 'huber':
            self.cost_func = huber_cost
        else:
            raise NotImplementedError(
                f'Cost type {config.lat_accel_type} not implemented for LATERAL ACCEL COST. '+
                'Please choose from [quadratic, huber]'
                )

    @partial(jax.jit, static_argnums=(0,))
    def get_running_cost(
			self, state: DeviceArray, ctrl: DeviceArray, ref: DeviceArray, time_idx: int
	) -> float:
        '''
        Given a state, control, and time index, return the cost.
        Input:
            state: (dim_x) state
            ctrl: (dim_u) control
            ref: (dim_ref) reference 
            time_idx: int (1)
        return:
            cost: float
        '''
        lat_accel = states[2, :]**2 / 0.257 * np.tan(states[-1, :])
        return jnp.sum(self.cost_func(ctrl, self.weight, self.delta))