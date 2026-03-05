from typing import List, Tuple

DEFAULT_IMG_SIZE_HW: Tuple[int, int] = (180, 320)  # (H, W)

# Data
DATA_ROOT = "train3"
IMG_EXT = ".jpg"
CSV_REQUIRED_COLUMNS: List[str] = ["pic_number", "x", "y", "rotation", "zoom"]

# Reference image pattern per sequence: seqX_start.jpg
REF_SUFFIX = "_start"

# Model / training
IN_CHANNELS = 6
OUT_DIM = 4

# Tanh output scaling: x/y are pixel displacements from reference center,
# so scale by half image dims. Tune rotation and zoom to your real data range.
USE_TANH_BOUNDING = True
# x/y are absolute pixel coords in the original camera frame before resize.

# Native camera resolution (labels are in this pixel space)
CAM_W, CAM_H, CAM_FPS = 1920, 1080, 25  # change to 3840, 2160, 25 for 4K

# Tanh output scaling derived from camera resolution 
# no manual tuning needed: OUT_SCALE = (CAM_W / 2, CAM_H / 2, 180.0, 2.0)
OUT_SCALE = (810.0, 433.0, 180.0, 2.0)

# Inference host: Raspberry Pi 5 + AI HAT+
# UDP output target (motor-control listener). If control is on same Pi, use 127.0.0.1.
UDP_TARGET_IP = "127.0.0.1"
UDP_TARGET_PORT = 5005
SEND_INTERVAL_S = 0.05  # ~20Hz; adjust as needed

# Camera (adjust per your Pi camera / USB camera)
DEVICE_PATH = "/dev/video0"


