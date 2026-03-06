# All SAM3 detections in one image frame

std_msgs/Header header

# text prompts used for this inference
string[] text_prompts

# all detections (bbox, 3D position, mask, class, score)
sam3_msgs/Detection[] detections
