from typing import Tuple, Optional, Dict
import time
import copy
import numpy as np
from functools import partial
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray

from .dynamics import Bicycle5D
from .cost import Cost
from .path import Path


class iLQR():

	def __init__(
			self, id: str, config, dyn: Bicycle5D, cost: Cost, path: Path,
			**kwargs
	) -> None:
		self.id = id
		self.config = config
		self.dyn = copy.deepcopy(dyn)
		self.cost = copy.deepcopy(cost)
		self.path = copy.deepcopy(path)

		# iLQR parameters
		self.dim_x = dyn.dim_x
		self.dim_u = dyn.dim_u
		self.N = config.N
		self.max_iter = config.MAX_ITER
		self.tol = 1e-3  # ILQR update tolerance.
		self.eps = getattr(config, "EPS", 1e-6)  # Numerical issue for Quu inverse.

		# Stepsize scheduler.
		self.alphas = 0.9**(np.arange(30))
		self.horizon_indices = jnp.arange(self.N).reshape(1, -1)

	def get_action(
			self, controls: Optional[np.ndarray] = None, **kwargs
	) -> np.ndarray:
		status = 0

		# `controls` include control input at timestep N-1, which is a dummy
		# control of zeros.
		if controls is None:
			controls = jnp.zeros((self.dim_u, self.N))
		else:
			assert controls.shape[1] == self.N
			controls = jnp.array(controls)

		# Rolls out the nominal trajectory and gets the initial cost.
		#* This is differnet from the naive iLQR as it relies on the information
		#* from the pyspline.
		states, controls = self.rollout_nominal(
				jnp.array(kwargs.get('state')), controls
		)
		closest_pt, slope, theta = self.path.get_closest_pts(
				np.asarray(states[:2, :])
		)
		closest_pt = jnp.array(closest_pt)
		slope = jnp.array(slope)
		theta = jnp.array(theta)
		J = self.cost.get_traj_cost(
				states, controls, closest_pt, slope, theta,
				time_indices=self.horizon_indices
		)

		converged = False
		time0 = time.time()
		for i in range(self.max_iter):
			# We need cost derivatives from 0 to N-1, but we only need dynamics
			# jacobian from 0 to N-2.
			c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
					states, controls, closest_pt, slope, theta,
					time_indices=self.horizon_indices
			)
			fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
			K_closed_loop, k_open_loop = self.backward_pass(
					c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu
			)
			updated = False
			for alpha in self.alphas:
				X_new, U_new, J_new, closest_pt_new, slope_new, theta_new = (
						self.forward_pass(
								states, controls, K_closed_loop, k_open_loop, alpha
						)
				)

				if J_new < J:  # Improved!
					if np.abs((J-J_new) / J) < self.tol:  # Small improvement.
						converged = True

					# Updates nominal trajectory and best cost.
					J = J_new
					states = X_new
					controls = U_new
					closest_pt = closest_pt_new
					slope = slope_new
					theta = theta_new
					updated = True
					break

			# Terminates early if there is no update within alphas.
			if not updated:
				status = 2
				break

			# Terminates early if the objective improvement is negligible.
			if converged:
				status = 1
				break
		t_process = time.time() - time0

		solver_info = dict(
				states=np.asarray(states), controls=np.asarray(controls),
				K_closed_loop=np.asarray(K_closed_loop),
				k_open_loop=np.asarray(k_open_loop), t_process=t_process,
				status=status, J=J, c_x=np.asarray(c_x), c_u=np.asarray(c_u),
				c_xx=np.asarray(c_xx), c_uu=np.asarray(c_uu), c_ux=np.asarray(c_ux),
				fx=np.asarray(fx), fu=np.asarray(fu)
		)
		return np.asarray(controls[:, 0]), solver_info

	def forward_pass(
			self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
			K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
	) -> Tuple[DeviceArray, DeviceArray, float, DeviceArray, DeviceArray,
						 DeviceArray]:
		X, U = self.rollout(
				nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
		)
		#* This is differnet from the naive iLQR as it relies on the information
		#* from the pyspline. Thus, it cannot be used with jit.
		closest_pt, slope, theta = self.path.get_closest_pts(np.asarray(X[:2, :]))
		closest_pt = jnp.array(closest_pt)
		slope = jnp.array(slope)
		theta = jnp.array(theta)
		J = self.cost.get_traj_cost(
				X, U, closest_pt, slope, theta, time_indices=self.horizon_indices
		)
		return X, U, J, closest_pt, slope, theta

	@partial(jax.jit, static_argnums=(0,))
	def rollout(
			self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
			K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
	) -> Tuple[DeviceArray, DeviceArray]:

		@jax.jit
		def _rollout_step(i, args):
			X, U = args
			u_fb = jnp.einsum(
					"ik,k->i", K_closed_loop[:, :, i], (X[:, i] - nominal_states[:, i])
			)
			u = nominal_controls[:, i] + alpha * k_open_loop[:, i] + u_fb
			x_nxt, u_clip = self.dyn.integrate_forward_jax(X[:, i], u)
			X = X.at[:, i + 1].set(x_nxt)
			U = U.at[:, i].set(u_clip)
			return X, U

		X = jnp.zeros((self.dim_x, self.N))
		U = jnp.zeros((self.dim_u, self.N))  #  Assumes the last ctrl are zeros.
		X = X.at[:, 0].set(nominal_states[:, 0])

		X, U = jax.lax.fori_loop(0, self.N - 1, _rollout_step, (X, U))
		return X, U

	@partial(jax.jit, static_argnums=(0,))
	def rollout_nominal(
			self, initial_state: DeviceArray, controls: DeviceArray
	) -> Tuple[DeviceArray, DeviceArray]:

		@jax.jit
		def _rollout_nominal_step(i, args):
			X, U = args
			x_nxt, u_clip = self.dyn.integrate_forward_jax(X[:, i], U[:, i])
			X = X.at[:, i + 1].set(x_nxt)
			U = U.at[:, i].set(u_clip)
			return X, U

		X = jnp.zeros((self.dim_x, self.N))
		X = X.at[:, 0].set(initial_state)
		X, U = jax.lax.fori_loop(
				0, self.N - 1, _rollout_nominal_step, (X, controls)
		)
		return X, U

	@partial(jax.jit, static_argnums=(0,))
	def backward_pass(
			self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
			c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray
	) -> Tuple[DeviceArray, DeviceArray]:
		"""
		Jitted backward pass looped computation.

		Args:
				c_x (DeviceArray): (dim_x, N)
				c_u (DeviceArray): (dim_u, N)
				c_xx (DeviceArray): (dim_x, dim_x, N)
				c_uu (DeviceArray): (dim_u, dim_u, N)
				c_ux (DeviceArray): (dim_u, dim_x, N)
				fx (DeviceArray): (dim_x, dim_x, N-1)
				fu (DeviceArray): (dim_x, dim_u, N-1)

		Returns:
				Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
				ks (DeviceArray): gain vectors (dim_u, N - 1)
		"""

		@jax.jit
		def backward_pass_looper(i, _carry):
			V_x, V_xx, ks, Ks = _carry
			n = self.N - 2 - i

			Q_x = c_x[:, n] + fx[:, :, n].T @ V_x
			Q_u = c_u[:, n] + fu[:, :, n].T @ V_x
			Q_xx = c_xx[:, :, n] + fx[:, :, n].T @ V_xx @ fx[:, :, n]
			Q_ux = c_ux[:, :, n] + fu[:, :, n].T @ V_xx @ fx[:, :, n]
			Q_uu = c_uu[:, :, n] + fu[:, :, n].T @ V_xx @ fu[:, :, n]

			Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)

			Ks = Ks.at[:, :, n].set(-Q_uu_inv @ Q_ux)
			ks = ks.at[:, n].set(-Q_uu_inv @ Q_u)

			V_x = Q_x + Q_ux.T @ ks[:, n]
			V_xx = Q_xx + Q_ux.T @ Ks[:, :, n]

			return V_x, V_xx, ks, Ks

		# Initializes.
		Ks = jnp.zeros((self.dim_u, self.dim_x, self.N - 1))
		ks = jnp.zeros((self.dim_u, self.N - 1))
		V_x = c_x[:, -1]
		V_xx = c_xx[:, :, -1]
		reg_mat = self.eps * jnp.eye(self.dim_u)

		V_x, V_xx, ks, Ks = jax.lax.fori_loop(
				0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
		)
		return Ks, ks
