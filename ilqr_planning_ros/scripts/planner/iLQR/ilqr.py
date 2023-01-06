from typing import Tuple, Optional, Dict
import time
import copy
import numpy as np
from functools import partial
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray

from .dynamics import Bicycle5D
from .cost import Cost, CollisionChecker, Obstacle
from .path import Path
from .config import Config
import time 

class iLQR():
	def __init__(self, config_file = None) -> None:

		self.config = Config()  # Load default config.
		if config_file is not None:
			self.config.load_config(config_file)  # Load config from file.
		
		# print(self.config)

		self.dyn = Bicycle5D(self.config)
		self.cost = Cost(self.config)
		self.collision_checker = CollisionChecker(self.config)
		self.obstacle_list = []
		self.path = None
		

		# iLQR parameters
		self.dim_x = self.config.num_dim_x
		self.dim_u = self.config.num_dim_u
		self.n = int(self.config.n)
		self.dt = float(self.config.dt)
		self.max_iter = int(self.config.max_iter)
		self.tol = float(self.config.tol)  # ILQR update tolerance.

		# line search parameters.
		self.alphas = self.config.line_search_a**(np.arange(0, 
                                                self.config.line_search_b,
                                                self.config.line_search_c)
                                            )
		# print(self.alphas)
		# regularization parameters
		self.reg_min = float(self.config.reg_min)
		self.reg_max = float(self.config.reg_max)
		self.reg_init = float(self.config.reg_init)
		self.reg_scale_up = float(self.config.reg_scale_up)
		self.reg_scale_down = float(self.config.reg_scale_down)

		self.warm_up()

	def update_path(self, path: Path):
		self.path = path

	def update_obstacles(self, vertices_list: list):
		self.obstacle_list = []
		for vertices in vertices_list:
			self.obstacle_list.append(Obstacle(vertices))

	def plan(self, init_state: np.ndarray, 
				controls: Optional[np.ndarray] = None) -> Dict:
		t_start = time.time()
		status = 0
		'''
		Main iLQR loop.
		Args:
			init_state: [num_dim_x] np.ndarray: initial state.
			control: [num_dim_u, N] np.ndarray: initial control.
		'''
		np.set_printoptions(suppress=True, precision=5)
		if controls is None:
			controls = jnp.zeros((self.dim_u, self.n))
		else:
			assert controls.shape[1] == self.n
			controls = jnp.asarray(controls)
		
		# Rolls out the nominal trajectory and gets the initial cost.
		#* This is differnet from the naive iLQR as it relies on the information
		#* from the pyspline.
		init_state = jnp.asarray(init_state, dtype=jnp.float32)
		states, controls = self.dyn.rollout_nominal(jnp.asarray(init_state), controls)
		refs = jnp.array(self.path.get_reference(states[:2, :]))
		obs_refs = self.collision_checker.check_collisions(states, self.obstacle_list)
		J = self.cost.get_traj_cost(states, controls, refs, obs_refs)
		converged = False
		reg = self.reg_init
		

		for _ in range(self.max_iter):
			updated = False
			# We need cost derivatives from 0 to N-1, but we only need dynamics
			# jacobian from 0 to N-2.
			c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(states, controls, refs, obs_refs)

			# We only need dynamics derivatives (A and B matrics) from 0 to N-2.
			fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
			
			# print("get_jacobian: ", time.time() - t0)
			while not updated and reg <= self.reg_max:
				# Backward pass with new regularization.
				K_closed_loop, k_open_loop = self.backward_pass(
					c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu, reg = reg
				)
				
				# Line search through alphas.
				for alpha in self.alphas:
					X_new, U_new, J_new, refs_new, obs_refs_new = (
							self.forward_pass(
									states, controls, K_closed_loop, k_open_loop, alpha
							)
					)
					
					if J_new < J:  # Improved!
						# Small improvement.
						# if np.abs(J-J_new)/max(1, np.abs(J)) < self.tol:
						if np.abs(J-J_new) < self.tol:
							converged = True
						print("Update from ", J, " to ", J_new, "reg: ", reg,
								"alpha: {0:0.3f}".format(alpha), "{0:0.3f}".format(time.time()-t_start))
						# Updates nominal trajectory and best cost.
						J = J_new
						states = X_new
						controls = U_new
						refs = refs_new
						obs_refs = obs_refs_new
						
						updated = True
						reg = max(reg*self.reg_scale_down, self.reg_min)
						break # break the for loop
						
				# If the line search fails, increase the regularization.
				if not updated:
					reg *= self.reg_scale_up

			# if no line search succeeds, terminate.
			if not updated:
				status = 2
				break
			# Terminates early if the objective improvement is negligible.
			if converged:
				status = 1
				break
			
		t_process = time.time() - t_start
		# print(obs_refs[:,-1,:])
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
		# t0 = time.time()
		refs = jnp.array(self.path.get_reference(X[:2, :]))
		# t1 = time.time()
		obs_refs = self.collision_checker.check_collisions(X, self.obstacle_list)
		# t2 = time.time()
		J = self.cost.get_traj_cost(X, U, refs, obs_refs)
		# # print("Cost time: ", t1-t0, t2-t1, time.time()-t2)

		return X, U, J, refs, obs_refs

	@partial(jax.jit, static_argnums=(0,))
	def backward_pass(
			self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
			c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray,
			reg: float
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

			# According to the paper, the regularization is added to Vxx for robustness.
			# http://roboticexplorationlab.org/papers/iLQR_Tutorial.pdf
			V_xx_reg = V_xx + reg_mat
			Q_ux_reg = c_ux[:, :, n] + fu[:, :, n].T @ V_xx_reg @ fx[:, :, n]
			Q_uu_reg = c_uu[:, :, n] + fu[:, :, n].T @ V_xx_reg @ fu[:, :, n]
			Q_uu_reg_inv = jnp.linalg.inv(Q_uu_reg)
			# Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)

			Ks = Ks.at[:, :, n].set(-Q_uu_reg_inv @ Q_ux_reg)
			ks = ks.at[:, n].set(-Q_uu_reg_inv @ Q_u)

			V_x = Q_x + Ks[:,:,n].T @ Q_u + Q_ux.T @ ks[:, n] + Ks[:,:,n].T @ Q_uu @ ks[:, n] 
			V_xx = Q_xx +  Ks[:, :, n].T @ Q_ux+ Q_ux.T @ Ks[:, :, n] + Ks[:, :, n].T @ Q_uu @ Ks[:, :, n]

			# V_x = Q_x + Q_ux.T @ ks[:, n]
			# V_xx = Q_xx + Q_ux.T @ Ks[:, :, n]

			# jax.debug.print("{x} vs {x1} || {y} vs {y1}", x=V_x_temp.shape, x1 =V_x.shape, y=V_xx_temp.shape, y1 = V_xx.shape)

			return V_x, V_xx, ks, Ks

		# Initializes.
		Ks = jnp.zeros((self.dim_u, self.dim_x, self.n - 1))
		ks = jnp.zeros((self.dim_u, self.n - 1))
		V_x = c_x[:, -1]
		V_xx = c_xx[:, :, -1]
		# reg_mat = reg * jnp.eye(self.dim_u)
		reg_mat = reg* jnp.eye(self.dim_x)

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
			X = X.at[:, i + 1].set(x_nxt)
			U = U.at[:, i].set(u_clip)
			return X, U

		X = jnp.zeros((self.dim_x, self.n))
		U = jnp.zeros((self.dim_u, self.n))  #  Assumes the last ctrl are zeros.
		X = X.at[:, 0].set(nominal_states[:, 0])

		X, U = jax.lax.fori_loop(0, self.n - 1, _rollout_step, (X, U))

		X.at[3,:].set(
			jnp.mod(X[3, :] + jnp.pi, 2 * jnp.pi) - jnp.pi
		)
		return X, U

	def warm_up(self):
		"""Warm up the jitted functions."""
		
		# Build a fake path as a 1 meter radius circle.
		theta = np.linspace(0, 2 * jnp.pi, 100)
		centerline = np.zeros([2, 100])
		centerline[0,:] = 1 * np.cos(theta)
		centerline[1,:] = 1 * np.sin(theta)

		self.path = Path(centerline, 0.5, 0.5, True)

		# add obstacle
		obs = np.array([[0, 0, 0.5, 0.5], [1, 1.5, 1, 1.5]]).T
		obs_list = [[obs for _ in range(self.n)]]
		self.update_obstacles(obs_list)

		x_init = np.array([0.0, -1.0, 1, 0, 0])

		# import matplotlib.pyplot as plt
		plan = self.plan(x_init)
		# plt.plot(plan['states'][0,:], plan['states'][1,:])
		# print(f"Warm up takes {plan['t_process']} seconds.")
		self.path = None
		self.obstacle_list = []
	

