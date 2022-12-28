from abc import ABC, abstractmethod
from functools import partial
from typing import Union
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

class BaseCost(ABC):
	'''
	Base class for cost functions.
	'''
	def __init__(self):
		super().__init__()

	@abstractmethod
	def get_stage_cost(
			self, state: DeviceArray, ctrl: DeviceArray, ref: DeviceArray, time_idx: int
	) -> float:
		'''
		Abstract method for computing the cost at a single timestep.
		Input:
			state: (dim_x)
			ctrl: (dim_u)
			ref: (dim_ref)
			time_idx: int 
		return:
			cost: float
		'''
		raise NotImplementedError

	@partial(jax.jit, static_argnums=(0,))
	def get_cost(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the cost.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			cost: (N)
		'''
		return jax.vmap(self.get_stage_cost,
						in_axes=(1, 1, 1, 1))(states, ctrls, refs, time_indices)

	@partial(jax.jit, static_argnums=(0,))
	def get_traj_cost(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
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
		costs = jax.vmap(self.get_stage_cost,
						in_axes=(1, 1, 1, 1))(states, ctrls, refs, time_indices)
		return jnp.sum(costs).astype(float)

	@partial(jax.jit, static_argnums=(0,))
	def get_cx(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the jacobian of cost w.r.t the state.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			dc_dx: (dim_x, N)
		'''
		_cx = jax.jacfwd(self.get_stage_cost, argnums=0)
		return jax.vmap(_cx, in_axes=(1, 1, 1, 1),
					out_axes=1)(states, ctrls, refs, time_indices)

	@partial(jax.jit, static_argnums=(0,))
	def get_cu(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the jacobian of cost w.r.t the control.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			dc_du: (dim_u, N)
		'''

		_cu = jax.jacfwd(self.get_stage_cost, argnums=1)
		return jax.vmap(_cu, in_axes=(1, 1, 1, 1),
					out_axes=1)(states, ctrls, refs, time_indices)

	@partial(jax.jit, static_argnums=(0,))
	def get_cxx(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the hessian of cost w.r.t the state.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			d^2c_dx^2: (dim_x, dim_x, N)
		'''
		_cxx = jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=0), argnums=0)
		return jax.vmap(_cxx, in_axes=(1, 1, 1, 1),
					out_axes=2)(states, ctrls, refs, time_indices)

	@partial(jax.jit, static_argnums=(0,))
	def get_cuu(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the hessian of cost w.r.t the control.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			d^2c_du^2: (dim_u, dim_u, N)
		'''
		_cuu = jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=1), argnums=1)
		return jax.vmap(_cuu, in_axes=(1, 1, 1, 1),
					out_axes=2)(states, ctrls, refs, time_indices)

	@partial(jax.jit, static_argnums=(0,))
	def get_cux(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the hessian of cost w.r.t the control and state.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			d^2c_dux: (dim_u, dim_x, N)
		'''
		_cux = jax.jacfwd(jax.jacrev(self.get_stage_cost, argnums=1), argnums=0)
		return jax.vmap(_cux, in_axes=(1, 1, 1, 1),
					out_axes=2)(states, ctrls, refs, time_indices)

	@partial(jax.jit, static_argnums=(0,))
	def get_cxu(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		'''
		Given a state, control, and time index, return the hessian of cost w.r.t the control and state.
		Input:
			states: (dim_x, N) List of states
			ctrls: (dim_u, N) List of controls
			refs: (dim_ref, N) List of references
			time_indices: (1, N) List of time indices
		return:
			d^2c_dxu: (dim_x, dim_u, N)
		'''
		return self.get_cux(states, ctrls, refs, time_indices).T

	@partial(jax.jit, static_argnums=(0,))
	def get_derivatives(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray, time_indices: DeviceArray
	) -> DeviceArray:
		return (
				self.get_cx(states, ctrls, refs, time_indices),
				self.get_cu(states, ctrls, refs, time_indices),
				self.get_cxx(states, ctrls, refs, time_indices),
				self.get_cuu(states, ctrls, refs, time_indices),
				self.get_cux(states, ctrls, refs, time_indices),
		)

class BarrierCost(BaseCost):
	def __init__(
		self, clip_min: float, clip_max: float, q1: float,
		q2: float, cost: BaseCost
	):
		super().__init__()
		self.clip_min = clip_min
		self.clip_max = clip_max
		self.q1 = q1
		self.q2 = q2
		self.cost = cost

	def get_stage_cost(
		self, states: DeviceArray, ctrls: DeviceArray, time_idx: DeviceArray
	) -> DeviceArray:
		_cost = self.cost.get_stage_cost(states, ctrls, time_idx)
		return self.q1 * jnp.exp(
			self.q2 * jnp.clip(a=_cost, a_min=self.clip_min, a_max=self.clip_max)
		)

@partial(jax.jit, static_argnums=(0,))

def exp_linear_cost(
		y: Union[float, DeviceArray], a: Union[float, DeviceArray] = 1
    ) -> Union[float, DeviceArray]:
	'''
	Base Class of Exponential Linear Cost defined as following
	Take y = func(x, u, t) 
		c(x, u, t) = a*exp(y) if y <=0
		c(x, u, t) = a*y+a if y > 0
	
	Args:
		y: float or 1D array, value to be costed
		a: float or 1D array, coefficient of exponential linear cost
		
	'''
	return jnp.where(y <= 0, a * jnp.exp(y), a * y + a)

@jax.jit
def quadratic_cost(
		y: Union[float, DeviceArray], a: Union[float, DeviceArray] = 1,
		not_used = None
	) -> Union[float, DeviceArray]:
	'''
	Base Class of Quadratic Cost defined as following
	Take y = func(x, u, t) 
		c(x, u, t) = (a*y)^2 
	Args:
		y: float or 1D array, value to be costed
		a: float or 1D array, coefficient of quadratic cost
	'''	
	return (a * y) ** 2

@jax.jit
def huber_cost(
		y: Union[float, DeviceArray], a: Union[float, DeviceArray] = 1,
		delta: Union[float, DeviceArray] = 1
    ) -> Union[float, DeviceArray]:
	'''
    Base Class of Huber Cost defined as following
    Take y = func(x, u, t) 
		z = |a*y}
		c(x, u, t) = y**2 if |y| <= delta
		c(x, u, t) = delta*(2*|y|-delta) if |y| > delta

	Args:
		y: float or 1D array, value to be costed
		delta: float or 1D array, threshold of Huber Cost
	'''
	z = jnp.abs(a*y)
	return jnp.where(z <= delta, z**2, delta*(2*z-delta))