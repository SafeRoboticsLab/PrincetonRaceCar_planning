import numpy as np
from racecar_obs_detection.srv import GetFRS, GetFRSResponse
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

def frs_to_obstacle(frs_response: GetFRSResponse) -> list:
    """
    Convert FRS service response into a list of obstacles usable by ILQR.
    Each obstacle is a list of 2D polygons (one per timestep).
    """
    obstacles_list = []
    for frs in frs_response.FRS:  # List of SetArray
        vertices_list = []
        for frs_t in frs.set_list:  # List of polygons over time
            polygon = [[p.x, p.y] for p in frs_t.points]
            vertices_list.append(np.array(polygon))
        obstacles_list.append(vertices_list)
    return obstacles_list

def frs_to_msg(frs_response: GetFRSResponse) -> MarkerArray:
    """
    Convert FRS data to a MarkerArray for visualization in RViz.
    Each polygon becomes a LINE_STRIP marker.
    """
    marker_array = MarkerArray()
    if not frs_response.FRS:
        return marker_array

    for i, frs in enumerate(frs_response.FRS):
        for t, frs_t in enumerate(frs.set_list):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = f"frs_{i}"
            marker.id = t
            marker.action = Marker.ADD
            marker.type = Marker.LINE_STRIP

            marker.color.r = 204.0 / 255.0
            marker.color.g = 102.0 / 255.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            # Add points and loop back to first point
            marker.points = frs_t.points + [frs_t.points[0]]

            marker.scale.x = 0.01
            marker.lifetime = Duration(sec=0, nanosec=200_000_000)  # 0.2 seconds

            marker_array.markers.append(marker)

    return marker_array
