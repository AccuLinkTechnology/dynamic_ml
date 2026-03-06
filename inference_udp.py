"""
Usage:
  python3 inference_udp.py pt   live
  python3 inference_udp.py pt   udp
  python3 inference_udp.py hef  live
  python3 inference_udp.py hef  udp
"""

import sys
import time
import socket
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import (
    UDP_TARGET_IP, UDP_TARGET_PORT, SEND_INTERVAL_S,
    DEVICE_PATH, CAM_W, CAM_H,
    DEFAULT_IMG_SIZE_HW, IN_CHANNELS, OUT_DIM,
    USE_TANH_BOUNDING, OUT_SCALE,
)

PT_WEIGHTS = "/home/acculink/Documents/dynamic_ml/runs_train3/20260305_094954/best_model.pt"
HEF_PATH   = "/home/acculink/Documents/dynamic_ml/laser_net.hef"

transform = T.Compose([
    T.Resize(DEFAULT_IMG_SIZE_HW, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def threshold_frame(frame_bgr):
    """Convert to grayscale, apply adaptive threshold, return 3-channel BGR."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # back to 3ch for transform


def capture_ref(cap) -> torch.Tensor:
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not capture reference frame.")
    frame = threshold_frame(frame)
    return transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def load_pt_model():
    from model import LaserNetSimple
    model = LaserNetSimple(IN_CHANNELS, OUT_DIM,
                           use_tanh_bounding=USE_TANH_BOUNDING,
                           out_scale=OUT_SCALE)
    ckpt = torch.load(PT_WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[PT] Loaded: {PT_WEIGHTS}")
    return model


def run_pt(model, x_in):
    scale = torch.tensor(OUT_SCALE, dtype=torch.float32)
    with torch.no_grad():
        return (model(x_in)[0] * scale).tolist()


def load_hef_model():
    from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdevice = VDevice(params)
    infer_model = vdevice.create_infer_model(HEF_PATH)
    infer_model.set_batch_size(1)
    infer_model.input().set_format_type(FormatType.FLOAT32)
    infer_model.output().set_format_type(FormatType.FLOAT32)
    configured = infer_model.configure()
    print(f"[HEF] Loaded: {HEF_PATH}")
    return infer_model, configured


def run_hef(infer_model, configured, x_in):
    inp = x_in.numpy().astype(np.float32)
    out_buf = np.empty(infer_model.output().shape, dtype=np.float32)
    bindings = configured.create_bindings()
    bindings.input().set_buffer(inp)
    bindings.output().set_buffer(out_buf)
    configured.run([bindings], timeout_ms=1000)
    raw = out_buf.flatten().tolist()
    return [raw[i] * OUT_SCALE[i] for i in range(4)]

def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("pt", "hef") or sys.argv[2] not in ("live", "udp"):
        print("Usage: python3 inference_udp.py [pt|hef] [live|udp]")
        sys.exit(1)

    backend, mode = sys.argv[1], sys.argv[2]

    if backend == "pt":
        model = load_pt_model()
        infer_fn = lambda x: run_pt(model, x)
    else:
        infer_model, configured = load_hef_model()
        infer_fn = lambda x: run_hef(infer_model, configured, x)

    cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {DEVICE_PATH}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    input("Press Enter to capture reference frame...")
    ref = capture_ref(cap)
    print(f"[REF] Captured. Backend: {backend}, Mode: {mode}. Press 'q' to quit.\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if mode == "udp" else None
    target = (UDP_TARGET_IP, UDP_TARGET_PORT)
    last_send = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame = threshold_frame(frame)
            cur = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            x_in = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)

            x_px, y_px, rot, zoom = infer_fn(x_in)

            now = time.time()
            if last_send == 0.0 or (now - last_send) >= SEND_INTERVAL_S:
                if mode == "udp":
                    sock.sendto(f"{x_px},{y_px},{rot},{zoom}".encode(), target)
                else:
                    print(f"x={x_px:+.1f} y={y_px:+.1f} rot={rot:+.1f} zoom={zoom:+.3f}")
                last_send = now

            cv2.putText(frame, f"x={x_px:+.1f} y={y_px:+.1f} rot={rot:+.1f} zoom={zoom:+.3f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Inference", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sock:
            sock.close()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
