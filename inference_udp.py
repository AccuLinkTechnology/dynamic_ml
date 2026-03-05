import sys
import time
import socket
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNetSimple
from config import (
    UDP_TARGET_IP, UDP_TARGET_PORT, SEND_INTERVAL_S,
    DEVICE_PATH, CAM_W, CAM_H,
    DEFAULT_IMG_SIZE_HW, IN_CHANNELS, OUT_DIM,
    USE_TANH_BOUNDING, OUT_SCALE,
)

WEIGHTS = "/home/acculink/Documents/dynamic_ml/runs_train3/20260304_172025/best_model.pt"

transform = T.Compose([
    T.Resize(DEFAULT_IMG_SIZE_HW, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def capture_ref(cap) -> torch.Tensor:
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not capture reference frame.")
    return transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("live", "udp"):
        print("Usage: python3 inference_udp.py [live|udp]")
        sys.exit(1)

    mode = sys.argv[1]
    scale = torch.tensor(OUT_SCALE, dtype=torch.float32)

    model = LaserNetSimple(IN_CHANNELS, OUT_DIM, use_tanh_bounding=USE_TANH_BOUNDING, out_scale=OUT_SCALE)
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {WEIGHTS}")

    cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {DEVICE_PATH}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    print(f"Camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    input("Press Enter to capture reference frame...")
    ref = capture_ref(cap)
    print(f"[REF] Captured. Mode: {mode}. Press 'q' to quit.\n")

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

            cur = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            x_in = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)

            with torch.no_grad():
                pred = model(x_in)[0] * scale

            x_px, y_px, rot, zoom = pred.tolist()

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