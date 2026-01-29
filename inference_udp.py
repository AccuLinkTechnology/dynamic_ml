# inference_udp_bias_cancel.py
# BIAS-CANCEL VERSION - works with biased CNN by capturing baseline offset

import os
import time
import json
import socket
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from model import LaserNet


# -------------------------
# USER CONFIG
# -------------------------
PI_IP = "192.168.1.37"
PI_PORT = 5005
SEND_HZ = 25

DEVICE_PATH = "/dev/video0"
DATA_ROOT = "/home/nvidia/Documents/dynamic_ml/train2"
SEQ_PROFILE = "seq5"
W, H, FPS = 1920, 1080, 25

IMG_SIZE = (320, 180)
REF_SAVE_DIR = "refs_live"

# BIAS-CANCEL CONFIG
BASELINE_SAMPLES = 10  # average this many frames for stable baseline
DEADBAND_AZ = 0.05     # ignore errors smaller than this (degrees)
DEADBAND_EL = 0.03
SMOOTHING_ALPHA = 0.3  # low-pass filter: 0=no smoothing, 1=no filtering


# -------------------------
# PREPROCESS (must match training)
# -------------------------
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


def save_bgr_as_tga(bgr_frame, out_path: str):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil.save(out_path)


def load_ref_from_seq_folder() -> tuple[torch.Tensor, str]:
    ref_path = os.path.join(DATA_ROOT, SEQ_PROFILE, f"{SEQ_PROFILE}_start 1.tga")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Saved seq reference not found: {ref_path}")

    ref_img = Image.open(ref_path).convert("RGB")
    ref_tensor = transform(ref_img)  # CPU tensor [3,H,W]
    return ref_tensor, ref_path


def capture_ref_from_camera(cap) -> tuple[torch.Tensor, str]:
    # let exposure settle a tiny bit
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not capture reference frame from camera.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REF_SAVE_DIR, f"{ts}_{SEQ_PROFILE}_ref.tga")
    save_bgr_as_tga(frame, out_path)

    ref_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ref_tensor = transform(ref_img)  # CPU tensor [3,H,W]
    return ref_tensor, out_path


def capture_baseline(cap, ref, model, device, mean, std, num_samples=10):
    """
    Capture baseline offset by averaging predictions over several frames
    while pole is in 'good' alignment position.
    """
    print(f"\n{'='*60}")
    print("BASELINE CALIBRATION")
    print(f"{'='*60}")
    print(f"Capturing {num_samples} samples to establish baseline offset...")
    print("Keep pole STABLE in the position you want to return to.")
    time.sleep(2)
    
    baseline_samples_az = []
    baseline_samples_el = []
    
    for i in range(num_samples):
        ok, frame = cap.read()
        if not ok:
            continue
            
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur = transform(pil).to(device)
        
        # Compute diff and predict
        x = (cur - ref).unsqueeze(0)
        
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()
        
        pred = pred_norm * std + mean
        az = float(pred[0].item())
        el = float(pred[1].item())
        
        baseline_samples_az.append(az)
        baseline_samples_el.append(el)
        
        print(f"  Sample {i+1}/{num_samples}: Δaz={az:+.3f}, Δel={el:+.3f}")
        time.sleep(0.1)
    
    # Average to get baseline
    baseline_az = np.mean(baseline_samples_az)
    baseline_el = np.mean(baseline_samples_el)
    baseline_az_std = np.std(baseline_samples_az)
    baseline_el_std = np.std(baseline_samples_el)
    
    print(f"\n{'='*60}")
    print(f"BASELINE ESTABLISHED:")
    print(f"  Azimuth:   {baseline_az:+.3f}° ± {baseline_az_std:.3f}")
    print(f"  Elevation: {baseline_el:+.3f}° ± {baseline_el_std:.3f}")
    print(f"{'='*60}")
    print("This offset will be subtracted from all future predictions.")
    print("Control system will now try to maintain THIS position.\n")
    
    return baseline_az, baseline_el


def apply_deadband(value, deadband):
    """Apply deadband - return 0 if within threshold"""
    if abs(value) < deadband:
        return 0.0
    return value


def low_pass_filter(current, previous, alpha):
    """Simple exponential smoothing: alpha=0 -> all previous, alpha=1 -> all current"""
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


def main():
    # UDP socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pi_addr = (PI_IP, PI_PORT)
    print(f"UDP target: {PI_IP}:{PI_PORT}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- Load GLOBAL label stats (new pipeline) ----
    stats = torch.load("./runs_ref/20260125_235217_ref/label_stats.pt", 
                       map_location="cpu", weights_only=True)
    mean = stats["global"]["mean"]
    std = stats["global"]["std"]
    print("Using label_stats.pt (global mean/std)")
    print("mean:", mean.tolist(), "std:", std.tolist())
    print("Using SEQ_PROFILE:", SEQ_PROFILE)

    # ---- Load model (diff input is still 3 channels) ----
    model = LaserNet(in_channels=3).to(device)
    model.load_state_dict(torch.load("./runs_ref/20260125_235217_ref/laser_net_best.pt", 
                                     map_location=device, weights_only=True))
    model.eval()

    # ---- Open camera via V4L2 ----
    cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {DEVICE_PATH} with CAP_V4L2")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    # ---- Choose reference at startup ----
    print("\nReference selection at startup:")
    print(f"  [1] Load saved seq reference: {os.path.join(DATA_ROOT, SEQ_PROFILE, f'{SEQ_PROFILE}_start 1.tga')}")
    print("  [2] Capture NEW reference from live camera (saved separately)")
    choice = input("Choose 1 or 2 then press Enter: ").strip()

    if choice == "2":
        ref_cpu, ref_source = capture_ref_from_camera(cap)
        print("[REF] Captured and saved to:", ref_source)
    else:
        ref_cpu, ref_source = load_ref_from_seq_folder()
        print("[REF] Loaded saved seq reference:", ref_source)

    ref = ref_cpu.to(device)

    # ---- Warm-up ----
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE[1], IMG_SIZE[0]).to(device)
        _ = model(dummy)

    # ---- CAPTURE BASELINE (THE KEY STEP) ----
    baseline_az, baseline_el = capture_baseline(cap, ref, model, device, mean, std, 
                                                 num_samples=BASELINE_SAMPLES)

    print("\nStarting BIAS-CANCELED inference + UDP send.")
    print("Press 'r' to re-baseline (if pole drifts)")
    print("Press 'q' to quit\n")

    # UDP send timing
    send_period = (1.0 / SEND_HZ) if SEND_HZ and SEND_HZ > 0 else 0.0
    last_send_t = 0.0
    
    # Smoothing state
    prev_err_az = None
    prev_err_el = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            continue

        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n[USER] Re-baseline requested...")
            baseline_az, baseline_el = capture_baseline(cap, ref, model, device, mean, std,
                                                        num_samples=BASELINE_SAMPLES)
            prev_err_az = None  # reset smoothing
            prev_err_el = None
            continue

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur = transform(pil).to(device)

        # ---- DIFF INPUT ----
        x = (cur - ref).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()
        infer_ms = (time.time() - t0) * 1000.0

        # Un-normalize
        pred = pred_norm * std + mean
        delta_az = float(pred[0].item())
        delta_el = float(pred[1].item())

        # ---- BIAS CANCELLATION (CRITICAL) ----
        err_az_raw = delta_az - baseline_az
        err_el_raw = delta_el - baseline_el

        # ---- DEADBAND ----
        err_az_db = apply_deadband(err_az_raw, DEADBAND_AZ)
        err_el_db = apply_deadband(err_el_raw, DEADBAND_EL)

        # ---- LOW-PASS FILTER ----
        err_az = low_pass_filter(err_az_db, prev_err_az, SMOOTHING_ALPHA)
        err_el = low_pass_filter(err_el_db, prev_err_el, SMOOTHING_ALPHA)
        prev_err_az = err_az
        prev_err_el = err_el

        # ---- UDP SEND ----
        now = time.time()
        if send_period == 0.0 or (now - last_send_t) >= send_period:
            payload = {
                "t": now,
                "seq": SEQ_PROFILE,
                "err_az": err_az,    # CHANGED: sending ERROR not raw delta
                "err_el": err_el,    # CHANGED: sending ERROR not raw delta
                "delta_az": delta_az,  # raw for debugging
                "delta_el": delta_el,  # raw for debugging
                "infer_ms": infer_ms,
            }
            sock.sendto(json.dumps(payload).encode("utf-8"), pi_addr)
            last_send_t = now

        # Console output
        status = "MOVE" if (abs(err_az) > 0.001 or abs(err_el) > 0.001) else "HOLD"
        print(f"[{status}] raw=({delta_az:+.3f},{delta_el:+.3f}) "
              f"err=({err_az:+.3f},{err_el:+.3f}) {infer_ms:.1f}ms")

        # Debug overlay
        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  infer={infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"BASELINE: az={baseline_az:+.2f} el={baseline_el:+.2f}",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"RAW: Δaz={delta_az:+.3f}  Δel={delta_el:+.3f}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(frame, f"ERR: {err_az:+.3f}  {err_el:+.3f}",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                    (0, 255, 0) if status == "HOLD" else (0, 165, 255), 2)
        cv2.imshow("Bias-Canceled Inference + UDP", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("\nShutdown complete.")


if __name__ == "__main__":
    main()