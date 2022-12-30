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
	def get_running_cost(
			self, state: DeviceArray, ctrl: DeviceArray, ref: DeviceArray
	) -> float:
		'''
		Abstract method for computing the cost at a single timestep.
		Input:
			state: (dim_x)
			ctrl: (dim_u)
			ref: (dim_ref)
		return:
			cost: float
		'''
		raise NotImplementedError

	@partial(jax.jit, static_argnums=(0,))
	def get_traj_cost(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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
		running_costs = jax.vmap(self.get_running_cost,
						in_axes=(1, 1, 1))(states, ctrls, refs)
		terminal_cost = self.get_terminal_cost(refs)
		return jnp.sum(running_costs).astype(float) + terminal_cost

	# Note: We assume that the Cost is in the Lagrange problem form. 
	# That is the Terminal cost is 0 
	# A little bit abusing the notation, but it is convenient for the implementation,
	# We assume there might be a "state/control independent" terminal cost.
	# In this case, we need to careful convert it to the Lagrange form and add it to the running cost.
	# See progress cost of State Cost for an example.

	@classmethod
	@partial(jax.jit, static_argnums=(0,))
	def get_terminal_cost(
			self, ref: DeviceArray
	) -> float:
		return 0

	@partial(jax.jit, static_argnums=(0,))
	def get_cx(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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
		_cx = jax.jacfwd(self.get_running_cost, argnums=0)
		return jax.vmap(_cx, in_axes=(1, 1, 1),
					out_axes=1)(states, ctrls, refs)

	@partial(jax.jit, static_argnums=(0,))
	def get_cu(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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

		_cu = jax.jacfwd(self.get_running_cost, argnums=1)
		return jax.vmap(_cu, in_axes=(1, 1, 1),
					out_axes=1)(states, ctrls, refs)

	@partial(jax.jit, static_argnums=(0,))
	def get_cxx(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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
		_cxx = jax.jacfwd(jax.jacrev(self.get_running_cost, argnums=0), argnums=0)
		return jax.vmap(_cxx, in_axes=(1, 1, 1),
					out_axes=2)(states, ctrls, refs)

	@partial(jax.jit, static_argnums=(0,))
	def get_cuu(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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
		_cuu = jax.jacfwd(jax.jacrev(self.get_running_cost, argnums=1), argnums=1)
		return jax.vmap(_cuu, in_axes=(1, 1, 1),
					out_axes=2)(states, ctrls, refs)

	@partial(jax.jit, static_argnums=(0,))
	def get_cux(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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
		_cux = jax.jacfwd(jax.jacrev(self.get_running_cost, argnums=1), argnums=0)
		return jax.vmap(_cux, in_axes=(1, 1, 1),
					out_axes=2)(states, ctrls, refs)

	@partial(jax.jit, static_argnums=(0,))
	def get_cxu(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
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
		return self.get_cux(states, ctrls, refs).T

	@partial(jax.jit, static_argnums=(0,))
	def get_derivatives(
			self, states: DeviceArray, ctrls: DeviceArray, refs: DeviceArray
	) -> DeviceArray:
		return (
				self.get_cx(states, ctrls, refs),
				self.get_cu(states, ctrls, refs),
				self.get_cxx(states, ctrls, refs),
				self.get_cuu(states, ctrls, refs),
				self.get_cux(states, ctrls, refs),
		)

@jax.jit
def exp_linear_cost(
		y: Union[float, DeviceArray], a: Union[float, DeviceArray] = 1,
		b: Union[float, DeviceArray] = 1
    ) -> Union[float, DeviceArray]:
	'''
	Base Class of Exponential Linear Cost defined as following
	Take y = func(x, u, t) 
		z = a*y
		c(x, u, t) = b*exp(z) if z <=0
		c(x, u, t) = b*z+b if z > 0
	
	Args:
		y: float or 1D array, value to be costed
		a: float or 1D array, coefficient of exponential linear cost
		
	'''
	z = a * y
	return jnp.where(z <= 0, b * jnp.exp(z), b * z + b)

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