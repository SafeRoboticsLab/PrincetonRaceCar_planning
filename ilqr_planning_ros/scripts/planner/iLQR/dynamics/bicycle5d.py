from typing import Tuple, Any
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp

class Bicycle5D():

	def __init__(self, config: Any, action_space: np.ndarray) -> None:
		"""
		Implements the bicycle dynamics (for Princeton race car). The state is the
		center of the rear axis.

		Args:
			config (Any): an object specifies configuration.
			action_space (np.ndarray): action space.
		"""
		self.dt: float = config.dt  # time step for each planning step
		self.dim_u = 2  # [accel, omega].
		self.dim_x = 5  # [x, y, v, psi, delta].

		# load parameters
		self.wheelbase: float = config.wheelbase  # vehicle chassis length
		self.delta_min = config.delta_min
		self.delta_max = config.delta_max
		self.v_min = config.v_min
		self.v_max = config.v_max
		self.a_min = config.a_min
		self.a_max = config.a_max

		self.ctrl_space = jnp.array([[self.a_min, self.a_max],
									[self.delta_min, self.delta_max]])

	def integrate_forward(
		self, state: np.ndarray, control: np.ndarray, **kwargs
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Finds the next state of the vehicle given the current state and
		control input.

		Args:
			state (np.ndarray).
			control (np.ndarray).

		Returns:
			np.ndarray: next state.
			np.ndarray: clipped control.
		"""
		state_nxt, ctrl_clip = self.integrate_forward_jax(
			jnp.array(state), jnp.array(control)
		)
		return np.array(state_nxt), np.array(ctrl_clip)

	@partial(jax.jit, static_argnums=(0,))
	def integrate_forward_jax(
		self, state: DeviceArray, control: DeviceArray
	) -> Tuple[DeviceArray, DeviceArray]:
		"""Clips the control and computes one-step time evolution of the system.

		Args:
			state (DeviceArray): [x, y, v, psi, delta].
			control (DeviceArray): [accel, omega].

		Returns:
			DeviceArray: next state.
			DeviceArray: clipped control.
		"""
		# Clips the controller values between min and max accel and steer values.
		ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
		state_nxt = self._integrate_forward(state, ctrl_clip)
		state_nxt = state_nxt.at[3].set(
			jnp.mod(state_nxt[3] + jnp.pi, 2 * jnp.pi) - jnp.pi
		)
		return state_nxt, ctrl_clip

	@partial(jax.jit, static_argnums=(0,))
	def deriv(self, state: DeviceArray, control: DeviceArray) -> DeviceArray:
		""" Computes the continuous system dynamics x_dot = f(x, u).
			dx_k = v_k cos(psi_k)
			dy_k = v_k sin(psi_k)
			dv_k = u[0]_k
			dpsi_k = v_k tan(delta_k) / L
			ddelta_k = u[1]_k

		Args:
			state (DeviceArray): [x, y, v, psi, delta].
			control (DeviceArray): [accel, omega].

		Returns:
			DeviceArray: next state.
		"""
		deriv = jnp.zeros((self.dim_x,))
		deriv = deriv.at[0].set(state[2] * jnp.cos(state[3]))
		deriv = deriv.at[1].set(state[2] * jnp.sin(state[3]))
		deriv = deriv.at[2].set(control[0])
		deriv = deriv.at[3].set(state[2] * jnp.tan(state[4]) / self.wheelbase)
		deriv = deriv.at[4].set(control[1])
		return deriv

	@partial(jax.jit, static_argnums=(0,))
	def _integrate_forward(
		self, state: DeviceArray, control: DeviceArray
	) -> DeviceArray:
		""" Computes one-step time evolution of the system: x_+ = f(x, u).
		The discrete-time dynamics is as below:
			x_k+1 = x_k + v_k cos(psi_k) dt
			y_k+1 = y_k + v_k sin(psi_k) dt
			v_k+1 = v_k + u0_k dt
			psi_k+1 = psi_k + v_k tan(delta_k) / L dt
			delta_k+1 = delta_k + u1_k dt

		Args:
			state (DeviceArray): [x, y, v, psi, delta].
			control (DeviceArray): [accel, omega].

		Returns:
			DeviceArray: next state.
		"""
		state_nxt = self._integrate_forward_dt(state, control, self.dt)
		return state_nxt

	@partial(jax.jit, static_argnums=(0,))
	def _integrate_forward_dt(
		self, state: DeviceArray, control: DeviceArray, dt: float
	) -> DeviceArray:
		"""4th-order Runge-Kutta method.

		Args:
			state (DeviceArray): current state
			control (DeviceArray): current control
			dt (float): time horizon to intergrate

		Returns:
			DeviceArray: next state
		"""
		k1 = self.deriv(state, control)
		k2 = self.deriv(state + k1*dt/2, control)
		k3 = self.deriv(state + k2*dt/2, control)
		k4 = self.deriv(state + k3*dt, control)
		return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

	@partial(jax.jit, static_argnums=(0,))
	def get_jacobian(
		self, nominal_states: DeviceArray, nominal_controls: DeviceArray
	) -> Tuple[DeviceArray, DeviceArray]:
		"""
		Returns the linearized 'A' and 'B' matrix of the ego vehicle around
		nominal states and controls.

		Args:
			nominal_states (DeviceArray): states along the nominal trajectory.
			nominal_controls (DeviceArray): controls along the trajectory.

		Returns:
			DeviceArray: the Jacobian of the dynamics w.r.t. the state.
			DeviceArray: the Jacobian of the dynamics w.r.t. the control.
		"""
		_jac = jax.jacfwd(self._integrate_forward, argnums=[0, 1])
		jac = jax.jit(jax.vmap(_jac, in_axes=(1, 1), out_axes=(2, 2)))
		return jac(nominal_states, nominal_controls)

	def rollout_nominal(
			self, initial_state: DeviceArray, controls: DeviceArray
	) -> Tuple[DeviceArray, DeviceArray]:
		'''
		Rolls out the nominal trajectory Givent the controls and initial state.
		'''
		n = controls.shape[1]
		@jax.jit
		def _rollout_nominal_step(i, args):
			X, U = args
			x_nxt, u_clip = self.integrate_forward_jax(X[:, i], U[:, i])
			X = X.at[:, i + 1].set(x_nxt)
			U = U.at[:, i].set(u_clip)
			return X, U

		X = jnp.zeros((self.dim_x, n))
		X = X.at[:, 0].set(initial_state)
		X, U = jax.lax.fori_loop(
				0, n - 1, _rollout_nominal_step, (X, controls)
		)
		return X, U
