"""
Barebones inference + UDP (raw predictions only) for Raspberry Pi 5 + AI HAT+.

Key rules:
- No calibration offset subtraction
- No deadband
- No correction / PID / isotonic
- UDP payload is SIMPLE: "az,el,rot,dist" (CSV string)

Example payload:
  0.142,-0.083,0.000,5.000
"""

import os
import time
import socket
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNetSimple
from config import (
    UDP_TARGET_IP, UDP_TARGET_PORT, SEND_INTERVAL_S,
    DEVICE_PATH, CAM_W, CAM_H, CAM_FPS,
    DEFAULT_IMG_SIZE_HW, IN_CHANNELS, OUT_DIM,
    USE_TANH_BOUNDING, OUT_SCALE,
    DATA_ROOT, IMG_EXT, REF_SUFFIX,
)

transform = T.Compose([
    T.Resize(DEFAULT_IMG_SIZE_HW, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_ref_from_seq_folder(seq: str) -> torch.Tensor:
    ref_path = os.path.join(DATA_ROOT, seq, f"{seq}{REF_SUFFIX}{IMG_EXT}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference not found: {ref_path}")
    ref_img = Image.open(ref_path).convert("RGB")
    return transform(ref_img)


def capture_ref_from_camera(cap) -> torch.Tensor:
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not capture reference frame from camera.")
    ref_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(ref_img)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (UDP_TARGET_IP, UDP_TARGET_PORT)
    print(f"UDP target: {UDP_TARGET_IP}:{UDP_TARGET_PORT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Torch device:", device)

    model_dir = os.environ.get("MODEL_DIR", "")
    weights_path = os.environ.get("WEIGHTS", "") or (os.path.join(model_dir, "best_model.pt") if model_dir else "")

    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Provide weights via env WEIGHTS=/path/to/best_model.pt "
            "or MODEL_DIR=/path/to/run_dir containing best_model.pt"
        )

    model = LaserNetSimple(
        in_channels=IN_CHANNELS,
        out_dim=OUT_DIM,
        use_tanh_bounding=USE_TANH_BOUNDING,
        out_scale=OUT_SCALE,
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded weights: {weights_path}")

    cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {DEVICE_PATH}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    seq = os.environ.get("SEQ_PROFILE", "seq1")

    print("\nReference selection:")
    print("  [1] Load saved seq reference")
    print("  [2] Capture NEW reference from live camera")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "2":
        ref_cpu = capture_ref_from_camera(cap)
        print("[REF] Captured from camera.")
    else:
        ref_cpu = load_ref_from_seq_folder(seq)
        print(f"[REF] Loaded from seq folder: {seq}")

    ref = ref_cpu.to(device)

    last_send = 0.0
    print("\nStarting inference. Press 'q' to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cur = transform(pil).to(device)

            x = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)

            with torch.no_grad():
                pred = model(x)[0].detach().cpu().tolist()

            az, el, rot, dist = pred

            now = time.time()
            if last_send == 0.0 or (now - last_send) >= SEND_INTERVAL_S:
                payload = f"{az},{el},{rot},{dist}"
                sock.sendto(payload.encode("utf-8"), target)
                last_send = now

            cv2.putText(frame, f"{seq} az={az:+.3f} el={el:+.3f} rot={rot:+.3f} dist={dist:+.3f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Inference (raw UDP)", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()