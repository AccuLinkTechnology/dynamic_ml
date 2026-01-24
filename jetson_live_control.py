import json
import socket
import time
from dataclasses import dataclass

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNet  # your no-BN model


# -----------------------
# CONFIG
# -----------------------
@dataclass
class ControlConfig:
    pi_ip: str = "192.168.1.37"     
    pi_port: int = 5005

    seq_profile: str = "seq4"       

    # Control shaping
    gain_az: float = 0.5
    gain_el: float = 0.5

    clamp_az: float = 1.0           # hard clamp on raw predicted deltas (motor units)
    clamp_el: float = 1.0

    deadband_az: float = 0.05       # below this, treat as zero
    deadband_el: float = 0.05

    rate_limit_az: float = 0.5      # max command per cycle
    rate_limit_el: float = 0.5

    ema_alpha: float = 0.4          # smoothing on commands (0 disables if set to 1.0)
    loop_hz: float = 50.0           # target loop rate

    # Video
    show_debug: bool = True


CFG = ControlConfig()

# -----------------------
# PREPROCESS (must match training)
# -----------------------
_transform = T.Compose([
    T.Resize((320, 180)),  # keep consistent with training
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])

def preprocess_bgr_frame(frame_bgr):
    # frame_bgr: numpy array from OpenCV
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    x = _transform(pil.convert("RGB"))
    return x


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def apply_deadband(v, db):
    return 0.0 if abs(v) < db else v


def main():
    # ---- Load stats
    seq_stats = torch.load("seq_stats.pt", map_location="cpu")
    if CFG.seq_profile not in seq_stats:
        raise ValueError(f"seq_profile '{CFG.seq_profile}' not in seq_stats.pt")

    mean = seq_stats[CFG.seq_profile]["mean"]
    std = seq_stats[CFG.seq_profile]["std"]

    # ---- Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = LaserNet().to(device)
    model.load_state_dict(torch.load("laser_net_best.pt", map_location=device))
    model.eval()

    # Warm-up (reduces first-frame latency)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 320, 180).to(device)
        _ = model(dummy)

    # ---- UDP socket to Pi
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pi_addr = (CFG.pi_ip, CFG.pi_port)


    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Failed to open /dev/video0 via CAP_V4L2")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))


    if not cap.isOpened():
        raise RuntimeError("Could not open video capture. Set GST_PIPELINE or correct device index.")

    # ---- Control state
    last_cmd_az = 0.0
    last_cmd_el = 0.0

    period = 1.0 / CFG.loop_hz
    print("Starting loop at", CFG.loop_hz, "Hz")

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        # If frame looks wrong, convert:
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)


        if not ok:
            print("Frame grab failed")
            time.sleep(0.05)
            continue

        # Preprocess
        x = preprocess_bgr_frame(frame).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()  # [2] normalized

        # Un-normalize back to motor units
        pred = pred_norm * std + mean
        delta_az = float(pred[0].item())
        delta_el = float(pred[1].item())

        # Control shaping: deadband -> clamp -> gain -> rate limit -> EMA
        delta_az = apply_deadband(delta_az, CFG.deadband_az)
        delta_el = apply_deadband(delta_el, CFG.deadband_el)

        delta_az = clamp(delta_az, -CFG.clamp_az, CFG.clamp_az)
        delta_el = clamp(delta_el, -CFG.clamp_el, CFG.clamp_el)

        cmd_az = CFG.gain_az * delta_az
        cmd_el = CFG.gain_el * delta_el

        cmd_az = clamp(cmd_az, -CFG.rate_limit_az, CFG.rate_limit_az)
        cmd_el = clamp(cmd_el, -CFG.rate_limit_el, CFG.rate_limit_el)

        # EMA smoothing (optional)
        a = CFG.ema_alpha
        cmd_az = a * cmd_az + (1 - a) * last_cmd_az
        cmd_el = a * cmd_el + (1 - a) * last_cmd_el

        last_cmd_az, last_cmd_el = cmd_az, cmd_el

        # Send to Pi
        msg = {
            "t": time.time(),
            "seq": CFG.seq_profile,
            "delta_az": cmd_az,
            "delta_el": cmd_el,
        }
        sock.sendto(json.dumps(msg).encode("utf-8"), pi_addr)

        # Debug display/print
        if CFG.show_debug:
            cv2.putText(frame, f"cmd_az={cmd_az:+.3f} cmd_el={cmd_el:+.3f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Jetson Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Rate control
        dt = time.time() - t0
        sleep_time = period - dt
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
