# infer_camera_udp.py
#
# Jetson: SDI (via /dev/video0) -> OpenCV V4L2 capture -> preprocess (match training)
#        -> LaserNet inference -> un-normalize using seq_stats.pt -> UDP send to Raspberry Pi
#
# Usage:
#   1) Edit PI_IP and SEQ_PROFILE below
#   2) On Pi:   python3 pi_command_receiver.py
#   3) On Jetson: python3 infer_camera_udp.py
# Press 'q' to quit.

import time
import json
import socket

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNet


# -------------------------
# USER CONFIG
# -------------------------
PI_IP = "192.168.1.37"   # e.g. "192.168.1.50"
PI_PORT = 5005

DEVICE_PATH = "/dev/video0"
W, H, FPS = 1920, 1080, 25

# Choose ONE sequence profile for now (seq2/seq3/seq4)
SEQ_PROFILE = "seq3"

# Optional: send at camera rate or slower (set to 0 to send every frame)
SEND_HZ = 25  # e.g. 25, 10, etc.


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

    # Device for model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load per-sequence stats
    seq_stats = torch.load("seq_stats.pt", map_location="cpu")
    if SEQ_PROFILE not in seq_stats:
        raise ValueError(f"SEQ_PROFILE={SEQ_PROFILE} not in seq_stats: {list(seq_stats.keys())}")
    mean = seq_stats[SEQ_PROFILE]["mean"]
    std = seq_stats[SEQ_PROFILE]["std"]

    print("Using SEQ_PROFILE:", SEQ_PROFILE)
    print("mean:", mean.tolist(), "std:", std.tolist())
    print(f"UDP target: {PI_IP}:{PI_PORT}")

    # Load model
    model = LaserNet().to(device)
    model.load_state_dict(torch.load("laser_net_best.pt", map_location=device))
    model.eval()

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

    # Send-rate control
    send_period = (1.0 / SEND_HZ) if SEND_HZ and SEND_HZ > 0 else 0.0
    last_send_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            continue

        # If you see incorrect colors, uncomment:
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

        # Preprocess
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = transform(pil).unsqueeze(0).to(device)

        # Inference
        t0 = time.time()
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()  # [2]
        infer_ms = (time.time() - t0) * 1000.0

        # Un-normalize to motor units
        pred = pred_norm * std + mean
        az = float(pred[0].item())
        el = float(pred[1].item())

        # UDP send (optionally rate-limited)
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

        # Debug overlay
        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  infer={infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Δaz={az:+.3f}  Δel={el:+.3f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow("Inference + UDP (Jetson)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Optional console print (comment out if too spammy)
        print(f"infer {infer_ms:6.1f} ms | Δaz={az:+.3f}  Δel={el:+.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
