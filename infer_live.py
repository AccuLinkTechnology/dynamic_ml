# infer_live.py
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
from model import LaserNet

IMG_SIZE = (320, 180)   # MUST match training
REF_IMAGE_PATH = "train2/seq1/seq1_start 1.tga"   # adjust for live deployment
MODEL_PATH = "laser_net_best.pt"


def build_transform():
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = build_transform()

    # Load reference frame once
    ref_pil = Image.open(REF_IMAGE_PATH).convert("RGB")
    ref_tensor = transform(ref_pil).to(device)

    # Model expects 3 channels for DIFF
    model = LaserNet(in_channels=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur_tensor = transform(pil).to(device)

        diff = (cur_tensor - ref_tensor).unsqueeze(0)

        with torch.no_grad():
            y_norm = model(diff)[0].cpu()

        delta_az = float(y_norm[0].item())
        delta_el = float(y_norm[1].item())

        print(f"Δaz={delta_az:+.3f}, Δel={delta_el:+.3f}")

        # TODO send to Pi motor controller here

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    main()
