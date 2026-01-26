# infer_camera_udp.py

import time
import json
import socket
import os

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from _model import LaserNet


# -------------------------
# USER CONFIG
# -------------------------
PI_IP = "192.168.1.37"
PI_PORT = 5005

DEVICE_PATH = "/dev/video0"
W, H, FPS = 1920, 1080, 25

SEQ_PROFILE = "seq3"
DATA_ROOT = "/home/nvidia/Documents/kam_ml/train2"

SEND_HZ = 25


# -------------------------
# PREPROCESS (must match training)
# -------------------------
transform = T.Compose([
    T.Resize((320, 180)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


def main():
    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pi_addr = (PI_IP, PI_PORT)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- LOAD GLOBAL LABEL STATS (NEW PIPELINE) ----
    stats = torch.load("label_stats.pt", map_location="cpu")
    mean = stats["global"]["mean"]
    std  = stats["global"]["std"]

    print("Using SEQ_PROFILE:", SEQ_PROFILE)
    print("mean:", mean.tolist(), "std:", std.tolist())
    print(f"UDP target: {PI_IP}:{PI_PORT}")

    # ---- LOAD MODEL (3-channel diff input) ----
    model = LaserNet(in_channels=3).to(device)
    model.load_state_dict(torch.load("laser_net_best.pt", map_location=device))
    model.eval()

    # ---- LOAD REFERENCE FRAME ONCE ----
    ref_path = os.path.join(DATA_ROOT, SEQ_PROFILE, f"{SEQ_PROFILE}_start 1.tga")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    ref_img = Image.open(ref_path).convert("RGB")
    ref_tensor = transform(ref_img).to(device)

    # Open camera via V4L2
    cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {DEVICE_PATH} with CAP_V4L2")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    # Warm-up
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 320, 180).to(device)
        _ = model(dummy)

    print("Starting inference + UDP send. Press 'q' to quit.")

    send_period = (1.0 / SEND_HZ) if SEND_HZ and SEND_HZ > 0 else 0.0
    last_send_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            continue

        # If colors wrong:
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

        # Preprocess current frame
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur_tensor = transform(pil).to(device)

        # ---- DIFF INPUT ----
        diff = cur_tensor - ref_tensor
        x = diff.unsqueeze(0)

        # Inference
        t0 = time.time()
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()
        infer_ms = (time.time() - t0) * 1000.0

        # Un-normalize
        pred = pred_norm * std + mean
        az = float(pred[0].item())
        el = float(pred[1].item())

        # UDP send
        now = time.time()
        if send_period == 0.0 or (now - last_send_t) >= send_period:
            payload = {
                "t": now,
                "seq": SEQ_PROFILE,
                "az": az,
                "el": el,
                "infer_ms": infer_ms,
            }
            sock.sendto(json.dumps(payload).encode("utf-8"), pi_addr)
            last_send_t = now

        # Debug overlay (UNCHANGED)
        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  infer={infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Δaz={az:+.3f}  Δel={el:+.3f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow("Inference + UDP (Jetson)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(f"infer {infer_ms:6.1f} ms | Δaz={az:+.3f}  Δel={el:+.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
