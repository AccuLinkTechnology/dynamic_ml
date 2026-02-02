import os
import time
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import csv
from threading import Lock

from model_v2 import LaserNetSimple  # CHANGED: Use new model
from model_pid import create_isotonic_corrector  # CHANGED: Use isotonic (no glitches!)


# ---- CONFIG ----
DEVICE_PATH = "/dev/video0"
DATA_ROOT = "/home/nvidia/Documents/dynamic_ml/train2"   # for loading the seq start ref
SEQ_PROFILE = "seq5"                                    # pick what matches today
W, H, FPS = 1920, 1080, 25

# Training uses torchvision Resize((H,W)) == (180,320)
IMG_SIZE_HW = (180, 320)                                # (H, W) MUST match training

REF_SAVE_DIR = "refs_live"                              # saved separately (no overwrite)
MODEL_DIR = "./runs_v2/20260201_113633_fast"     # CHANGED: Your latest best run

# CSV logging
CSV_FILE = "model_predictions.csv"  # Output file for logged predictions
LOG_ENABLED = True  # Set to False to disable logging

# Correction settings
USE_CORRECTION = False  # Apply calibrated gain/offset correction
USE_PID = False  # Enable PID for residual error (set True for closed-loop)


# ---- Preprocess must match training ----
transform = T.Compose([
    T.Resize(IMG_SIZE_HW, interpolation=T.InterpolationMode.BILINEAR),  # CHANGED: Explicit interpolation
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


def save_bgr_as_tga(bgr_frame, out_path: str):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil.save(out_path)  # inferred from .tga extension


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- CSV Setup ----
    pic_number = 1
    csv_lock = Lock()
    
    if LOG_ENABLED:
        # Check if file exists to determine if we need header
        file_exists = os.path.isfile(CSV_FILE)
        csv_file = open(CSV_FILE, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        
        if not file_exists:
            csv_writer.writerow(["pic_number", "delta_azimuth", "delta_elevation"])
            print(f"Created new CSV: {CSV_FILE}")
        else:
            # Read last pic_number to continue sequence
            with open(CSV_FILE, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # More than just header
                    last_line = lines[-1].strip().split(',')
                    pic_number = int(last_line[0]) + 1
            print(f"Appending to existing CSV: {CSV_FILE} (starting at pic #{pic_number})")
        
        csv_file.flush()
    else:
        csv_file = None
        csv_writer = None
        print("CSV logging disabled")

    # ---- Load config (CHANGED: new format) ----
    config_path = os.path.join(MODEL_DIR, "config.pt")
    if os.path.exists(config_path):
        config = torch.load(config_path, map_location="cpu")
        out_scale = config.get("out_scale", (2.6, 2.6))
        print("Using:", config_path)
        print("Output scale:", out_scale)
    else:
        out_scale = (2.6, 2.6)
        print("Warning: config.pt not found, using default out_scale")

    # ---- Load model (CHANGED: new architecture, no normalization) ----
    model = LaserNetSimple(in_channels=6, out_scale=out_scale).to(device)
    weights_path = os.path.join(MODEL_DIR, "best_model.pt")  # CHANGED: filename
    
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'mae' in checkpoint:
            print(f"  Validation MAE: {checkpoint['mae']}")
        if 'bias' in checkpoint:
            print(f"  Validation BIAS: {checkpoint['bias']}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Loaded model:", weights_path)

    # ---- Open camera via V4L2 (KEEP THIS EXACTLY) ----
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

    # ---- Calibration: compute stable-frame offset (CHANGED: new calibration method) ----
    print("\nCalibrating on reference frame...")
    with torch.no_grad():
        # Stable input: ref compared to itself = zero motion
        x_calib = torch.cat([ref, ref - ref], dim=0).unsqueeze(0)  # [1,6,H,W]
        calib_offset = model(x_calib)[0].cpu()  # [2]
    
    print(f"Calibration offset: az={calib_offset[0].item():+.3f}, el={calib_offset[1].item():+.3f}")

    # ---- Initialize model corrector (NEW) ----
    corrector = create_isotonic_corrector()  # Isotonic = smooth, no glitches
    print(f"\nModel corrector initialized: ISOTONIC (smooth, continuous)")
    print(f"  Correction: {'ENABLED' if USE_CORRECTION else 'DISABLED'}")
    print(f"  PID: {'ENABLED' if USE_PID else 'DISABLED'}")

    # ---- Warm-up (reduces first-infer overhead) ----
    with torch.no_grad():
        dummy = torch.zeros(1, 6, IMG_SIZE_HW[0], IMG_SIZE_HW[1]).to(device)
        _ = model(dummy)

    print("\nStarting inference (CONCAT + calibration).")
    if LOG_ENABLED:
        print(f"Press ENTER to log current prediction to {CSV_FILE}")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            continue

        # If colors look wrong, uncomment:
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur = transform(pil).to(device)

        # ---- CONCAT INPUT ----
        # x = [cur, cur-ref] along channel dimension -> [6,H,W]
        x = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)  # CHANGED: cur-ref for diff channels

        t0 = time.time()
        with torch.no_grad():
            pred_raw = model(x)[0].cpu()  # [2]
        infer_ms = (time.time() - t0) * 1000.0

        # CHANGED: Apply calibration offset (no normalization needed)
        pred = pred_raw - calib_offset
        az_uncorrected = float(pred[0].item())
        el_uncorrected = float(pred[1].item())
        
        # NEW: Apply model correction
        if USE_CORRECTION:
            az, el = corrector.correct(
                az_uncorrected, 
                el_uncorrected,
                enable_pid=USE_PID,
                dt=0.02  # Assume ~50Hz
            )
        else:
            az, el = az_uncorrected, el_uncorrected

        print(f"infer {infer_ms:6.1f} ms | Raw=({az_uncorrected:+.3f},{el_uncorrected:+.3f}) | "
              f"Corrected=({az:+.3f},{el:+.3f}) | Pic #{pic_number}", end='\r')

        # Display frame
        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  infer={infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"REF={os.path.basename(ref_source)}",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show raw model output
        cv2.putText(frame, f"Raw: ({az_uncorrected:+.3f}, {el_uncorrected:+.3f})",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)  # Gray
        
        # Show corrected output (main value)
        cv2.putText(frame, f"Corrected: ({az:+.3f}, {el:+.3f})",
                    (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)  # Green
        
        # Show calibration offset
        cv2.putText(frame, f"CAL: az={calib_offset[0].item():+.2f} el={calib_offset[1].item():+.2f}",
                    (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Yellow
        
        # Show pic number
        cv2.putText(frame, f"Pic #{pic_number}",
                    (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if LOG_ENABLED:
            cv2.putText(frame, "Press ENTER to log",
                        (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Cyan
        
        cv2.imshow("Inference Preview (V2 + Calibration)", frame)

        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 13 and LOG_ENABLED:  # Enter key (ASCII 13)
            with csv_lock:
                csv_writer.writerow([pic_number, az, el])
                csv_file.flush()
                print(f"\n[LOGGED] Pic #{pic_number}: Δaz={az:+.3f}, Δel={el:+.3f}")
                pic_number += 1

    cap.release()
    cv2.destroyAllWindows()
    
    if LOG_ENABLED and csv_file:
        csv_file.close()
        print(f"\nCSV saved: {CSV_FILE}")
        print(f"Total predictions logged: {pic_number - 1}")


if __name__ == "__main__":
    main()