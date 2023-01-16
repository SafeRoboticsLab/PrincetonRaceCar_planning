import hppfcl
import numpy as np

class Obstacle:
    def __init__(self, time_varying_vertices) -> None:
        '''
        Constuct a time-varying obstacle from a list of vertices
        time_varying_vertices: LIST: a list of 2D numpy arrays of shape (n, 2)
        '''
        self.steps = len(time_varying_vertices)
        self.time_varying_obstacle = []
        for vertices in time_varying_vertices:
            self.time_varying_obstacle.append(self._construct_collision_object(vertices))        
        
    def _construct_collision_object(self, vertices):
        """
        Construct a collision object from a list of vertices
        vertices: a 2D numpy array of shape (n, 2)
        """
        num_points = vertices.shape[0]
        dim = vertices.shape[1]
        
        assert num_points >= 3, f"The number of vertices ({num_points}) must be at least 3"
        assert dim == 2, f"The dimension of the vertices ({dim}) must be 2"
        
        # add a column of ones to the polygon
        vertices = np.append(vertices, np.ones((num_points, 1)), axis=1)
        
        # create a polytope object
        verts = hppfcl.StdVec_Vec3f()
        verts.extend(list(vertices))
        vertices[:,-1] = -1
        verts.extend(list(vertices))
        
        polygon = hppfcl.Convex.convexHull(verts, False, None)
        
        return polygon 
    
    def at(self, step):
        assert step < self.steps, f"The step {step} is out of range [0, {self.steps-1}]"
        return self.time_varying_obstacle[step]