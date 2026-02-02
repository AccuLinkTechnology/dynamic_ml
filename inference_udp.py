# infer_udp_corrected.py
"""
Jetson inference with model correction + UDP send to Pi.
No PID on Pi side - all correction done here on Jetson.
"""

import os
import time
import json
import socket
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model_v2 import LaserNetSimple
from model_pid import create_isotonic_corrector  # Isotonic = no glitches


# ---- CONFIG ----
PI_IP = "192.168.1.37"
PI_PORT = 5005
SEND_INTERVAL = 16.0  # Send command every 6 seconds

DEVICE_PATH = "/dev/video0"
DATA_ROOT = "/home/nvidia/Documents/dynamic_ml/train2"
SEQ_PROFILE = "seq28"
W, H, FPS = 1920, 1080, 25

IMG_SIZE_HW = (180, 320)
REF_SAVE_DIR = "refs_live"
MODEL_DIR = "./runs_v2/20260201_113633_fast"

# Correction settings
USE_CORRECTION = True  # Apply isotonic correction
USE_PID = False  # PID disabled - Pi does direct motor control

# Deadband (applied on Jetson before sending)
DEADBAND_AZ = 0.03
DEADBAND_EL = 0.03


# ---- Preprocess ----
transform = T.Compose([
    T.Resize(IMG_SIZE_HW, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def save_bgr_as_tga(bgr_frame, out_path: str):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil.save(out_path)


def load_ref_from_seq_folder() -> tuple:
    ref_path = os.path.join(DATA_ROOT, SEQ_PROFILE, f"{SEQ_PROFILE}_start 1.tga")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Saved seq reference not found: {ref_path}")
    ref_img = Image.open(ref_path).convert("RGB")
    ref_tensor = transform(ref_img)
    return ref_tensor, ref_path


def capture_ref_from_camera(cap) -> tuple:
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not capture reference frame from camera.")
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REF_SAVE_DIR, f"{ts}_{SEQ_PROFILE}_ref.tga")
    save_bgr_as_tga(frame, out_path)
    
    ref_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ref_tensor = transform(ref_img)
    return ref_tensor, out_path


def apply_deadband(value, deadband):
    """Apply deadband - return 0 if within threshold"""
    if abs(value) < deadband:
        return 0.0
    return value


def main():
    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pi_addr = (PI_IP, PI_PORT)
    print(f"UDP target: {PI_IP}:{PI_PORT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load config
    config_path = os.path.join(MODEL_DIR, "config.pt")
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location="cpu")
        out_scale = config.get("out_scale", (2.6, 2.6))
    else:
        out_scale = (2.6, 2.6)

    # Load model
    model = LaserNetSimple(in_channels=6, out_scale=out_scale).to(device)
    weights_path = os.path.join(MODEL_DIR, "best_model.pt")
    checkpoint = torch.load(weights_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded.")

    # Initialize corrector
    corrector = create_isotonic_corrector()  # Isotonic = smooth, no glitches
    print(f"\nCorrection: {'ENABLED (isotonic)' if USE_CORRECTION else 'DISABLED'}")

    # Open camera
    cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {DEVICE_PATH}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    # Choose reference
    print("\nReference selection:")
    print(f"  [1] Load saved seq reference")
    print(f"  [2] Capture NEW reference from live camera")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "2":
        ref_cpu, ref_source = capture_ref_from_camera(cap)
        print(f"[REF] Captured: {ref_source}")
    else:
        ref_cpu, ref_source = load_ref_from_seq_folder()
        print(f"[REF] Loaded: {ref_source}")

    ref = ref_cpu.to(device)

    # Calibration
    print("\nCalibrating on reference frame...")
    with torch.no_grad():
        x_calib = torch.cat([ref, ref - ref], dim=0).unsqueeze(0)
        calib_offset = model(x_calib)[0].cpu()
    print(f"Calibration offset: az={calib_offset[0].item():+.3f}, el={calib_offset[1].item():+.3f}")

    # Warm-up
    with torch.no_grad():
        dummy = torch.zeros(1, 6, IMG_SIZE_HW[0], IMG_SIZE_HW[1]).to(device)
        _ = model(dummy)

    print("\nStarting inference + UDP send (every 6 seconds). Press 'q' to quit.\n")

    last_send_t = 0.0
    
    # Store latest predictions
    latest_az_send = 0.0
    latest_el_send = 0.0
    latest_az_raw = 0.0
    latest_el_raw = 0.0
    latest_infer_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur = transform(pil).to(device)

        # Inference
        x = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)
        
        t0 = time.time()
        with torch.no_grad():
            pred_raw = model(x)[0].cpu()
        infer_ms = (time.time() - t0) * 1000.0

        # Calibration
        pred = pred_raw - calib_offset
        az_raw = float(pred[0].item())
        el_raw = float(pred[1].item())

        # Correction
        if USE_CORRECTION:
            az, el = corrector.correct(az_raw, el_raw, enable_pid=USE_PID, dt=0.02)
        else:
            az, el = az_raw, el_raw

        # Deadband
        az_send = apply_deadband(az, DEADBAND_AZ)
        el_send = apply_deadband(el, DEADBAND_EL)
        
        # Update latest values
        latest_az_send = az_send
        latest_el_send = el_send
        latest_az_raw = az_raw
        latest_el_raw = el_raw
        latest_infer_ms = infer_ms

        # UDP send every 6 seconds
        now = time.time()
        time_since_send = now - last_send_t
        if last_send_t == 0.0 or time_since_send >= SEND_INTERVAL:
            payload = {
                "t": now,
                "seq": SEQ_PROFILE,
                "corrected_az": az_send,  # Pi expects these
                "corrected_el": el_send,
                "raw_az": az_raw,  # For debugging
                "raw_el": el_raw,
                "infer_ms": infer_ms,
            }
            sock.sendto(json.dumps(payload).encode("utf-8"), pi_addr)
            last_send_t = now
            print(f"\n[SENT] corrected=({az_send:+.3f},{el_send:+.3f}) at t={now:.1f}")

        # Status
        status = "MOVE" if (abs(az_send) > 0.001 or abs(el_send) > 0.001) else "HOLD"
        next_send_in = SEND_INTERVAL - time_since_send if last_send_t > 0 else 0
        
        print(f"[{status}] raw=({az_raw:+.3f},{el_raw:+.3f}) → "
              f"send=({az_send:+.3f},{el_send:+.3f}) | "
              f"next send in {next_send_in:.1f}s | {infer_ms:.1f}ms", end='\r')

        # Display
        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  {latest_infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Raw: ({latest_az_raw:+.2f}, {latest_el_raw:+.2f})",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.putText(frame, f"Send: ({latest_az_send:+.2f}, {latest_el_send:+.2f})",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if status == "HOLD" else (0, 165, 255), 2)
        
        # Send countdown
        next_send_in = max(0, SEND_INTERVAL - time_since_send) if last_send_t > 0 else SEND_INTERVAL
        cv2.putText(frame, f"Next send: {next_send_in:.1f}s",
                    (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Jetson Inference + Correction (6s interval)", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("\n\nShutdown complete.")


if __name__ == "__main__":
    main()