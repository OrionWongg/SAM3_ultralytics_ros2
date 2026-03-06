# One SAM3 detection result

# class label
int32 class_id
string class_name

# detection confidence [0, 1]
float64 score

# 2D bounding box in inference-image pixel coordinates
sam3_msgs/BoundingBox2D bbox

# 3D position (filled by depth_estimator_node; empty/zero if depth unavailable)
sam3_msgs/BoundingBox3D bbox3d

# segmentation mask info
sam3_msgs/Mask mask
