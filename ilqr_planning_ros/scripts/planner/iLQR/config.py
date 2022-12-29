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
        self.max_iter = 10 # maximum number of iterations
        self.tol = 1e-3 # tolerance for the iLQR convergence
        
        ####################################################
        ############### Dynamics Parameters ################
        ####################################################
        self.wheelbase = 0.257 # wheelbase of the vehicle
        
        # steering angle limits
        self.delta_max = 0.35 # maximum steering angle
        self.delta_min = -0.35 # minimum steering angle
        
        # velocity limits
        self.v_max = 10.0 # maximum velocity
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
        self.path_weight = 1. # weight for the path deviation cost
        self.path_huber_delta = 1. # huber loss delta for path deviation cost
        
        # Velocity Cost
        self.dim_vel_ref = 3 # dimension of reference velocity in the reference
        self.vel_cost_type = 'quadratic' # 'quadratic' or 'huber'
        self.vel_weight = 1. # weight for the velocity cost
        self.vel_huber_delta = 1. # huber loss delta for velocity cost
        
        ########        Control Cost          ############
        self.ctrl_cost_type = 'quadratic' # 'quadratic' or 'huber'
        self.ctrl_cost_weight = [0.1,0.1]
        self.ctrl_cost_huber_delta = [1.,1.] # huber loss delta
        
        
        
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        