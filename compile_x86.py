import numpy as np
import glob
from hailo_sdk_client import ClientRunner

ONNX  = "runs_train3/20260305_094954/best_model.onnx"
NAME  = "laser_net"
ARCH  = "hailo8"
H, W, C = 180, 320, 6

runner = ClientRunner(hw_arch=ARCH)
runner.translate_onnx_model(ONNX, NAME,
    net_input_shapes={"input": [1, C, H, W]})
runner.save_har(f"{NAME}_parsed.har")

#calibration data: at some point get this from inference on real frames versus generated data.
calib_files = sorted(glob.glob("calib_data/*.npy"))[:200]
calib = np.concatenate([np.load(f) for f in calib_files], axis=0).astype(np.float32)
print(f"Calibration: {calib.shape[0]} frames")

runner.load_har(f"{NAME}_parsed.har")
runner.optimize(calib)
runner.save_har(f"{NAME}_quantized.har")

runner.load_har(f"{NAME}_quantized.har")
runner.compile()
runner.save_hef(f"{NAME}.hef")
print("Done: laser_net.hef")