# inference_udp.py
import socket
import torch
from PIL import Image
import torchvision.transforms as T
from model import LaserNet

IMG_SIZE = (320, 180)
REF_IMAGE_PATH = "train2/seq1/seq1_start 1.tga"
MODEL_PATH = "laser_net_best.pt"

UDP_IP = "192.168.1.50"   # Raspberry Pi IP
UDP_PORT = 5005


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

    # Load reference once
    ref_pil = Image.open(REF_IMAGE_PATH).convert("RGB")
    ref_tensor = transform(ref_pil).to(device)

    model = LaserNet(in_channels=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        # Replace with your camera capture pipeline
        frame = get_current_frame_somehow()  

        pil = Image.fromarray(frame)
        cur_tensor = transform(pil).to(device)
        diff = (cur_tensor - ref_tensor).unsqueeze(0)

        with torch.no_grad():
            y = model(diff)[0].cpu()

        delta_az = float(y[0].item())
        delta_el = float(y[1].item())

        msg = f"{delta_az:.4f},{delta_el:.4f}"
        sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))
        print("Sent:", msg)


if __name__ == "__main__":
    main()
