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
from .config import Config
import time 

class iLQR():

	def __init__(self, config_file = None) -> None:

		self.config = Config()  # Load default config.
		if config_file is not None:
			self.config.load(config_file)  # Load config from file.
		
		self.dyn = Bicycle5D(self.config)
		self.cost = Cost(self.config)
		self.path = None
		

		# iLQR parameters
		self.dim_x = self.config.num_dim_x
		self.dim_u = self.config.num_dim_u
		self.n = self.config.n
		self.dt = self.config.dt
		self.max_iter = self.config.max_iter
		self.tol = 1e-3  # ILQR update tolerance.

		# line search parameters.
		self.alphas = self.config.line_search_a**(np.arange(0, 
                                                self.config.line_search_a,
                                                self.config.line_search_c)
                                            )

		# regularization parameters
		self.reg_min = self.config.reg_min
		self.reg_max = self.config.reg_max
		self.reg_scale_up = self.config.reg_scale_up
		self.reg_scale_down = self.config.reg_scale_down
		self.reg_init = self.config.reg_init
		self.horizon_indices = jnp.arange(self.n).reshape(1, -1)

	def update_path(self, path: Path):
		self.path = path

	def plan(self, init_state: np.ndarray, 
				controls: Optional[np.ndarray] = None) -> Dict:
		status = 0
		'''
		Main iLQR loop.
		Args:
			init_state: [num_dim_x] np.ndarray: initial state.
			control: [num_dim_u, N] np.ndarray: initial control.
		'''

		if controls is None:
			controls = jnp.zeros((self.dim_u, self.n))
		else:
			assert controls.shape[1] == self.n
			controls = jnp.array(controls)

		# Rolls out the nominal trajectory and gets the initial cost.
		#* This is differnet from the naive iLQR as it relies on the information
		#* from the pyspline.
		states, controls = self.dyn.rollout_nominal(jnp.array(init_state), controls)

		refs = jnp.array(self.path.get_reference(states[:2, :]))

		# t0 = time.time()
		J = self.cost.get_traj_cost(states, controls, refs, self.horizon_indices).block_until_ready()
		# print("get traj cost time: ", time.time() - t0)

		converged = False
		reg = self.reg_init
		time0 = time.time()
		for _ in range(self.max_iter):
			# print("Iteration: ", i)
			# We need cost derivatives from 0 to N-1, but we only need dynamics
			# jacobian from 0 to N-2.
			c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
								states, controls, refs, self.horizon_indices)
			# We on;y need dynamics derivatives (A and B matrics) from 0 to N-2.
			fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
			K_closed_loop, k_open_loop = self.backward_pass(
					c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu, reg = reg
			)
			updated = False
			for alpha in self.alphas:
				# print("alpha: ", alpha)
				X_new, U_new, J_new, refs_new = (
						self.forward_pass(
								states, controls, K_closed_loop, k_open_loop, alpha
						)
				)
				if J_new < J:  # Improved!
					# print("updated to ", J_new, " from ", J , " with alpha ", alpha, " and reg ", reg)
					if np.abs((J-J_new) / J) < self.tol:  # Small improvement.
						converged = True

					# Updates nominal trajectory and best cost.
					J = J_new
					states = X_new
					controls = U_new
					refs = refs_new
					updated = True
					reg = max(self.reg_min, reg*self.reg_scale_down)

					break

			# Terminates early if there is no update within alphas.
			if not updated:
				reg *= self.reg_scale_up

			if reg > self.reg_max:
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
				fx=np.asarray(fx), fu=np.asarray(fu), N = self.n, dt = self.dt
		)
		return solver_info

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

		refs = jnp.array(self.path.get_reference(X[:2, :]))
		J = self.cost.get_traj_cost(X, U, refs, self.horizon_indices)
		return X, U, J, refs

	@partial(jax.jit, static_argnums=(0,))
	def backward_pass(
			self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
			c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray,
			reg: float = 1e-6
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
			n = self.n - 2 - i

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
		Ks = jnp.zeros((self.dim_u, self.dim_x, self.n - 1))
		ks = jnp.zeros((self.dim_u, self.n - 1))
		V_x = c_x[:, -1]
		V_xx = c_xx[:, :, -1]
		reg_mat = reg * jnp.eye(self.dim_u)

		V_x, V_xx, ks, Ks = jax.lax.fori_loop(
				0, self.n - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
		)
		return Ks, ks


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
			# jax.debug.print("{u}, {u_clip}", u = u, u_clip = u_clip)
			X = X.at[:, i + 1].set(x_nxt)
			U = U.at[:, i].set(u_clip)
			return X, U

		X = jnp.zeros((self.dim_x, self.n))
		U = jnp.zeros((self.dim_u, self.n))  #  Assumes the last ctrl are zeros.
		X = X.at[:, 0].set(nominal_states[:, 0])

		X, U = jax.lax.fori_loop(0, self.n - 1, _rollout_step, (X, U))
		return X, U
