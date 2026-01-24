import torch
from PIL import Image
import numpy as np
import cv2

from model import LaserNet
from dataset2 import LaserDataset  # only for consistent preprocessing if you want

DATA_ROOT = "/home/nvidia/Documents/kam_ml"
SEQ_PROFILE = "seq2"   # <-- set this for the session you’re running

def preprocess_pil(pil_img, img_size=(320, 180)):
    # Same as dataset2’s transform but inline to avoid constructing a Dataset
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

    seq_stats = torch.load("seq_stats.pt", map_location="cpu")
    mean = seq_stats[SEQ_PROFILE]["mean"]
    std = seq_stats[SEQ_PROFILE]["std"]

    model = LaserNet().to(device)
    model.load_state_dict(torch.load("laser_net_best.pt", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)  # replace with your SDI capture pipeline as needed

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = preprocess_pil(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            y_norm = model(x)[0].cpu()

        # Un-normalize back to motor units
        y = y_norm * std + mean
        delta_az = float(y[0].item())
        delta_el = float(y[1].item())

        print(f"Δaz={delta_az:+.3f}, Δel={delta_el:+.3f}")

        # TODO: send to Raspberry Pi here
        # send_motor_command(delta_az, delta_el)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

if __name__ == "__main__":
    main()
