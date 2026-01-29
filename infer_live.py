import os
import time
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from model import LaserNet


# ---- CONFIG ----
DEVICE_PATH = "/dev/video0"
DATA_ROOT = "/home/nvidia/Documents/dynamic_ml/train2"   # for loading the seq start ref
SEQ_PROFILE = "seq5"                                    # pick what matches today
W, H, FPS = 1920, 1080, 25

# Training uses torchvision Resize((H,W)) == (180,320)
IMG_SIZE_HW = (180, 320)                                # (H, W) MUST match training

REF_SAVE_DIR = "refs_live"                              # saved separately (no overwrite)
MODEL_DIR = "./runs_ref/20260129_194854_ref"            # <-- your new best run folder


# ---- Preprocess must match training ----
transform = T.Compose([
    T.Resize(IMG_SIZE_HW),
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

    # ---- Load label stats ----
    stats_path = os.path.join(MODEL_DIR, "label_stats.pt")
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["global"]["mean"]
    std = stats["global"]["std"]
    print("Using:", stats_path)
    print("mean:", mean.tolist(), "std:", std.tolist())

    # Optional visibility: baseline used for this seq during training (if present)
    if "baseline_by_seq" in stats and SEQ_PROFILE in stats["baseline_by_seq"]:
        b = stats["baseline_by_seq"][SEQ_PROFILE]
        if isinstance(b, torch.Tensor):
            b = b.tolist()
        print(f"training baseline_by_seq[{SEQ_PROFILE}] =", b)

    # ---- Load model (CONCAT => 6 channels) ----
    model = LaserNet(in_channels=6).to(device)
    weights_path = os.path.join(MODEL_DIR, "laser_net_best.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
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

    # ---- Warm-up (reduces first-infer overhead) ----
    with torch.no_grad():
        dummy = torch.zeros(1, 6, IMG_SIZE_HW[0], IMG_SIZE_HW[1]).to(device)  # [1,6,H,W]
        _ = model(dummy)

    print("\nStarting inference (CONCAT). Press 'q' to quit.")

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
        # x = [cur, ref] along channel dimension -> [6,H,W]
        x = torch.cat([cur, ref], dim=0).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            pred_norm = model(x)[0].cpu()  # [2]
        infer_ms = (time.time() - t0) * 1000.0

        pred = pred_norm * std + mean
        az = float(pred[0].item())
        el = float(pred[1].item())

        print(f"infer {infer_ms:6.1f} ms | Δaz={az:+.3f}  Δel={el:+.3f}")

        cv2.putText(frame, f"SEQ={SEQ_PROFILE}  infer={infer_ms:.1f}ms",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"REF={os.path.basename(ref_source)}",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Δaz={az:+.3f}  Δel={el:+.3f}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow("Inference Preview (CONCAT)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
