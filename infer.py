import time
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNet

# ---- CONFIG ----
DEVICE_PATH = "/dev/video0"
SEQ_PROFILE = "seq3"   # choose seq2/seq3/seq4 (pick what matches today)
W, H, FPS = 1920, 1080, 25

# ---- Preprocess must match training ----
transform = T.Compose([
    T.Resize((320, 180)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load stats
    seq_stats = torch.load("seq_stats.pt", map_location="cpu")
    if SEQ_PROFILE not in seq_stats:
        raise ValueError(f"SEQ_PROFILE={SEQ_PROFILE} not in seq_stats: {list(seq_stats.keys())}")

    mean = seq_stats[SEQ_PROFILE]["mean"]
    std  = seq_stats[SEQ_PROFILE]["std"]
    print("Using SEQ_PROFILE:", SEQ_PROFILE)
    print("mean:", mean.tolist(), "std:", std.tolist())

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

    # Warm-up (reduces first-infer overhead)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 320, 180).to(device)
        _ = model(dummy)

    print("Starting inference. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            continue

        # If colors look wrong, uncomment:
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

        # OpenCV gives BGR, convert to RGB for PIL
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = transform(pil).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()   # shape [2]
        infer_ms = (time.time() - t0) * 1000.0

        pred = pred_norm * std + mean
        az = float(pred[0].item())
        el = float(pred[1].item())

        print(f"infer {infer_ms:6.1f} ms | Δaz={az:+.3f}  Δel={el:+.3f}")

        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  infer={infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Δaz={az:+.3f}  Δel={el:+.3f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow("Inference Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
