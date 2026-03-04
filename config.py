from typing import List, Tuple

DEFAULT_IMG_SIZE_HW: Tuple[int, int] = (180, 320)  # (H, W)

# Data
DATA_ROOT = "train3"
IMG_EXT = ".jpg"
CSV_REQUIRED_COLUMNS: List[str] = ["pic_number", "x", "y", "rotation", "distance"]

# Reference image pattern per sequence: seqX_start 1.jpg
REF_SUFFIX = "_start 1"

# Model / training
IN_CHANNELS = 6
OUT_DIM = 4

# Optional safety bounding (default ON with neutral scales)
USE_TANH_BOUNDING = True
OUT_SCALE = (1.0, 1.0, 1.0, 1.0)

# Inference host: Raspberry Pi 5 + AI HAT+
# UDP output target (motor-control listener). If control is on same Pi, use 127.0.0.1.
UDP_TARGET_IP = "127.0.0.1"
UDP_TARGET_PORT = 5005
SEND_INTERVAL_S = 0.05  # ~20Hz; adjust as needed

# Camera (adjust per your Pi camera / USB camera)
DEVICE_PATH = "/dev/video0"
CAM_W, CAM_H, CAM_FPS = 1920, 1080, 25
