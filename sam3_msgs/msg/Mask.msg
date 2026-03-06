# Segmentation mask for one detected instance

# size of the inference image the mask was generated from
int32 height
int32 width

# mask area in pixels
int32 area

# boundary contour of the mask (polygon points, may be empty if not computed)
sam3_msgs/Point2D[] contour
