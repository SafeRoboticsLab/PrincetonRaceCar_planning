from typing import Tuple, Optional, Dict, Union
from jaxlib.xla_extension import DeviceArray
import time
import os
import numpy as np
import jax
from .dynamics import Bicycle5D
from .cost import Cost, CollisionChecker, Obstacle
from .ref_path import RefPath
from .config import Config
import time

status_lookup = ["Iteration Limit Exceed",
                "Converged",
                "Failed Line Search"]

class iLQRnp():
	def __init__(self, config_file = None) -> None:

		self.config = Config()  # Load default config.
		if config_file is not None:
			self.config.load_config(config_file)  # Load config from file.

		print("iLQR setting:", self.config)

		# Set up Jax parameters
		jax.config.update('jax_platform_name', self.config.platform)
		print("Jax using Platform: ", jax.lib.xla_bridge.get_backend().platform)

		# If you want to use GPU, lower the memory fraction from 90% to avoid OOM.
		os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "20"

		self.dyn = Bicycle5D(self.config)
		self.cost = Cost(self.config)
		self.ref_path = None

		# iLQR parameters
		self.dim_x = self.config.num_dim_x
		self.dim_u = self.config.num_dim_u
		self.T = int(self.config.T)
		self.dt = float(self.config.dt)
		self.max_iter = int(self.config.max_iter)
		self.tol = float(self.config.tol)  # ILQR update tolerance.

		# line search parameters.
		self.alphas = self.config.line_search_base**(np.arange(self.config.line_search_a,
                                                self.config.line_search_b,
                                                self.config.line_search_c)
                                            )

		print("Line Search Alphas: ", self.alphas)

		# regularization parameters
		self.reg_min = float(self.config.reg_min)
		self.reg_max = float(self.config.reg_max)
		self.reg_init = float(self.config.reg_init)
		self.reg_scale_up = float(self.config.reg_scale_up)
		self.reg_scale_down = float(self.config.reg_scale_down)
		self.max_attempt = self.config.max_attempt

		# collision checker
		# Note: This will not be used until lab2.
		self.collision_checker = CollisionChecker(self.config)
		self.obstacle_list = []

		self.warm_up()

	def warm_up(self):
		"""Warm up the jitted functions."""

		# Build a fake path as a 1 meter radius circle.
		theta = np.linspace(0, 2 * np.pi, 100)
		centerline = np.zeros([2, 100])
		centerline[0,:] = 1 * np.cos(theta)
		centerline[1,:] = 1 * np.sin(theta)

		self.ref_path = RefPath(centerline, 0.5, 0.5, 1, True)

		# add obstacle
		obs = np.array([[0, 0, 0.5, 0.5], [1, 1.5, 1, 1.5]]).T
		obs_list = [[obs for _ in range(self.T)]]
		self.update_obstacles(obs_list)

		x_init = np.array([0.0, -1.0, 1, 0, 0])
		print("Start warm up iLQR...")
		# import matplotlib.pyplot as plt
		self.plan(x_init, verbose=False)
		print("iLQR warm up finished.")
		# plt.plot(plan['trajectory'][0,:], plan['trajectory'][1,:])
		# print(f"Warm up takes {plan['t_process']} seconds.")
		self.ref_path = None
		self.obstacle_list = []

	def update_ref_path(self, ref_path: RefPath):
		'''
		Update the reference path.
		Args:
			ref_path: RefPath: reference path.
		'''
		self.ref_path = ref_path

	def update_obstacles(self, vertices_list: list):
		'''
		Update the obstacle list for a list of vertices.
		Args:
			vertices_list: list of np.ndarray: list of vertices for each obstacle.
		'''
		# Note: This will not be used until lab2.
		self.obstacle_list = []
		for vertices in vertices_list:
			self.obstacle_list.append(Obstacle(vertices))

	def get_references(self, trajectory: Union[np.ndarray, DeviceArray]):
		'''
		Given the trajectory, get the path reference and obstacle information.
		Args:
			trajectory: [num_dim_x, T] trajectory.
		Returns:
			path_refs: [num_dim_x, T] np.ndarray: references.
			obs_refs: [num_dim_x, T] np.ndarray: obstacle references.
		'''
		trajectory = np.asarray(trajectory)
		path_refs = self.ref_path.get_reference(trajectory[:2, :])
		obs_refs = self.collision_checker.check_collisions(trajectory, self.obstacle_list)
		return path_refs, obs_refs

	def plan(self, init_state: np.ndarray,
				controls: Optional[np.ndarray] = None, verbose=False) -> Dict:
		'''
		Main iLQR loop.
		Args:
			init_state: [num_dim_x] np.ndarray: initial state.
			control: [num_dim_u, T] np.ndarray: initial control.
		'''

		# We first check if the planner is ready
		if self.ref_path is None:
			# TODO: define your own return behavior in case there is no reference path.
			print("No reference path is provided.")
			return dict(status=-1)

		# if no initial control sequence is provided, we assume it is all zeros.
		if controls is None:
			controls =np.zeros((self.dim_u, self.T))
		else:
			assert controls.shape[1] == self.T

		# Start timing
		t_start = time.time()

		# Rolls out the nominal trajectory and gets the initial cost.
		trajectory, controls = self.dyn.rollout_nominal_np(init_state, controls)
		# Get path and obstacle references.
		path_refs, obs_refs = self.get_references(trajectory)
		# Get the initial cost of the trajectory.
		J = self.cost.get_traj_cost(trajectory, controls, path_refs, obs_refs)

		##########################################################################
		# TODO: Implement the iLQR algorithm. Feel free to add any helper functions.
		# You will find following implemented functions useful:

		'''
		A, B = self.dyn.get_jacobian_np(trajectory, controls)

		Returns the linearized 'A' and 'B' matrix of the ego vehicle around
		nominal trajectory and controls.

		Args:
			trajectory (DeviceArray): trajectory along the nominal trajectory.
			controls (DeviceArray): controls along the trajectory.

		Returns:
			np.ndarray: the Jacobian of the dynamics w.r.t. the state.
			np.ndarray: the Jacobian of the dynamics w.r.t. the control.
		'''
		'''
		state_next, control_clip = self.dyn.integrate_forward_np(state, control)
		
		Finds the next state of the vehicle given the current state and
		control input.

		Args:
			state:(np.ndarray) [5].
			control: (np.ndarray) [2].

		Returns:
			state_next: np.ndarray: next state. [5]
			control_clip: np.ndarray: clipped control. [2]
		'''

		'''
		q, r, Q, R, H = self.cost.get_derivatives_np(trajectory, controls, path_refs, obs_refs)
		
		Given a state, control, and references, return Jacobians and Hessians of cost function
		Input:
			trajectory: (dim_x, N) array of state trajectory
			controls: (dim_u, N) array of control sequence
			path_refs: (dim_ref, N) array of references (e.g. reference path, reference velocity, etc.)
			obs_refs: -Optional- (num_obstacle, 2, N) List of obstacles
		return:
			q: np.ndarray, jacobian of cost function w.r.t. trajectory
            r: np.ndarray, jacobian of cost function w.r.t. controls
            Q: np.ndarray, hessian of cost function w.r.t. trajectory
            R: np.ndarray, hessian of cost function w.r.t. controls
            H: np.ndarray, hessian of cost function w.r.t. trajectory and controls
		'''
		##########################################################################

		t_process = time.time() - t_start
		solver_info = dict(
				t_process=t_process, # Time spent on planning
				N = self.T, dt = self.dt,
				trajectory = trajectory,
				controls = controls,
				status=None, #	TODO: Fill this in
				K_closed_loop=None, # TODO: Fill this in
				k_open_loop=None # TODO: Fill this in
				# Optional TODO: Fill in other information you want to return
		)
		return solver_info



