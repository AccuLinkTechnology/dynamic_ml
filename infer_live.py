import torch
from PIL import Image
import numpy as np
import cv2
import os

from _model import LaserNet

DATA_ROOT = "/home/nvidia/Documents/kam_ml/train2"
SEQ_PROFILE = "seq2"   # which reference image to use
IMG_SIZE = (320, 180)

def preprocess_pil(pil_img, img_size=IMG_SIZE):
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5])
    ])
    return transform(pil_img.convert("RGB"))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- load global label stats (new pipeline) ----
    stats = torch.load("label_stats.pt", map_location="cpu")
    mean = stats["global"]["mean"]
    std  = stats["global"]["std"]

    # ---- load model (3-channel diff input) ----
    model = LaserNet(in_channels=3).to(device)
    model.load_state_dict(torch.load("laser_net_best.pt", map_location=device))
    model.eval()

    # ---- load reference frame once ----
    ref_path = os.path.join(DATA_ROOT, SEQ_PROFILE, f"{SEQ_PROFILE}_start 1.tga")
    ref_img = Image.open(ref_path).convert("RGB")
    ref_tensor = preprocess_pil(ref_img).to(device)

    cap = cv2.VideoCapture(0)  # replace with SDI pipeline if needed

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur_tensor = preprocess_pil(pil).to(device)

        # ---- DIFF INPUT ----
        diff = cur_tensor - ref_tensor
        x = diff.unsqueeze(0)

        with torch.no_grad():
            y_norm = model(x)[0].cpu()

        # Un-normalize to motor units
        y = y_norm * std + mean
        delta_az = float(y[0].item())
        delta_el = float(y[1].item())

        print(f"Δaz={delta_az:+.3f}, Δel={delta_el:+.3f}")

        # TODO: send to Raspberry Pi
        # send_motor_command(delta_az, delta_el)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    main()
