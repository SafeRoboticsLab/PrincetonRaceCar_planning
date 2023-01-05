import yaml

class Config():
    def __init__(self):

        ####################################################
        ############### General Parameters #################
        ####################################################
        self.num_dim_x = 5
        self.num_dim_u = 2
        self.n = 10 # horizon length
        self.dt = 0.1 # time step

        
        ####################################################
        ###########Optimization Parameters #################
        ####################################################        
        self.max_iter = 100 # maximum number of iterations
        # tolerance for the iLQR convergence
        # Make sure this is smaller than the minimum line search step size
        self.tol = 1e-3
        
        # line search parameters
        # We assume line search parameter is a**np.arange(0, b, c)
        self.line_search_a = 0.1 # line search parameter a
        self.line_search_b = 5 # line search parameter b
        self.line_search_c = 1 # line search parameter c
        
        # regularization parameters
        self.reg_min = 1e-3 # minimum regularization
        self.reg_max = 1e5 # maximum regularization
        self.reg_scale_down = 0.1 # scale down factor for regularization
        self.reg_scale_up = 10.0 # scale up factor for regularization
        self.reg_init = 1.0 # initial regularization
        
        ####################################################
        ############### Dynamics Parameters ################
        ####################################################
        self.wheelbase = 0.257 # wheelbase of the vehicle
        self.radius = 0.13 # radius of the vehicle
        self.width = 0.22 # width of the vehicle
        self.length = 0.40 # length of the vehicle
        
        # steering angle limits
        self.delta_max = 0.35 # maximum steering angle
        self.delta_min = -0.35 # minimum steering angle
        
        # velocity limits
        self.v_max = 5.0 # maximum velocity
        self.v_min = 0.0 # minimum velocity
        
        # turn rate limits
        self.omega_min = -6.0 # minimum turn rate
        self.omega_max = 6.0 # maximum turn rate
        
        # acceleration limits
        self.a_max = 5.0 # maximum acceleration
        self.a_min = -5.0 # minimum acceleration
        
        ####################################################
        ########## Parameters for iLQR COST ################
        ####################################################
        
        ########        State Cost          ############
        
        # Path Offset Cost
        self.dim_closest_pt_x = 0 # dimension of closest point x in the reference
        self.dim_closest_pt_y = 1 # dimension of closest point y in the reference
        self.dim_path_slope = 2 # dimension of path slope in the reference
        self.path_cost_type = 'quadratic' # 'quadratic' or 'huber'
        self.path_weight = 10.0 # weight for the path deviation cost
        self.path_huber_delta = 2 # huber loss delta for path deviation cost
        
        # Velocity Cost
        self.dim_vel_ref = 3 # dimension of reference velocity in the reference
        self.vel_cost_type = 'quadratic' # 'quadratic' or 'huber'
        self.vel_weight = 1.0 # weight for the velocity cost
        self.vel_huber_delta = 0.5 # huber loss delta for velocity cost
        
        # Lateral Acceleration Cost
        # We use ExpLinearCost for lateral acceleration cost
        self.lat_accel_thres = 6.0 # threshold for lateral acceleration cost
        self.lat_accel_a = 5.0 # parameter for lateral acceleration cost
        self.lat_accel_b = 1.0 # parameter for ExpLinear Cost
        
        # Progress Cost
        self.dim_progress = 4 # dimension of progress in the reference
        self.progress_weight = 2.0  # weight for the progress cost
        
        ########        Control Cost          ############
        
        self.ctrl_cost_type = 'quadratic' # 'quadratic' or 'huber'
        # those value should not be too small
        self.ctrl_cost_accel_weight = 1
        self.ctrl_cost_steer_weight = 1
        self.ctrl_cost_accel_huber_delta = 1.0 # huber loss delta
        self.ctrl_cost_steer_huber_delta = 1.0 # huber loss delta
        
        ########        Obstacle Cost          ############
        self.dim_obs_x = 5 # dimension of obstacle x in the reference
        self.dim_obs_y = 6 # dimension of obstacle y in the reference
        self.dim_obs_radius = 7 # dimension of obstacle radius in the reference
        
        self.obs_a = 10.0 # parameter for obstacle cost
        self.obs_b = 15.0 # parameter for ExpLinear Cost
                
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            
        config_varible = vars(self)
        for key in config_varible.keys():
            if key in config_dict:
                setattr(self, key, config_dict[key])
                
    def __str__(self) -> str:
        return 'iLQR config: ' + str(vars(self))