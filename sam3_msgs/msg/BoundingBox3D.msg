# 3D position of the bounding box center in camera frame

# 3D position in meters (camera coordinate system: X right, Y down, Z forward)
geometry_msgs/Point position

# depth at bbox center (meters); -1.0 if measurement is invalid
float64 depth_m

# frame reference (e.g. "camera_color_optical_frame")
string frame_id
