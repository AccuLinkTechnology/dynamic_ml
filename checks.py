from PIL import Image
import matplotlib.pyplot as plt
from dataset import preprocess
import pandas as pd

img = Image.open("./seq2/seq2_0 1.tga")

def check1():


    print("Mode:", img.mode)
    print("Size:", img.size)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

def check2():
    # Use the same image as Step 1

    x = preprocess(img)

    print("Tensor shape:", x.shape)  # should be [3, H, W]

    # Convert back to displayable image
    vis = (x.permute(1, 2, 0).numpy() + 1.0) / 2.0

    plt.imshow(vis)
    plt.title("What the network sees")
    plt.axis("off")
    plt.show()

def check3():
    DATA_ROOT = "/home/nvidia/Documents/kam_ml"
    SEQ = "seq2"

    df = pd.read_csv(f"{DATA_ROOT}/{SEQ}.csv")

    # Pick ONE row to inspect
    row = df.iloc[27]

    pic_num = int(row["pic_number"])
    delta_az = row["delta_azimuth"]
    delta_el = row["delta_elevation"]

    print(f"Inspecting image: {SEQ}_{pic_num} 1.tga")
    print(f"Label: Δaz = {delta_az}, Δel = {delta_el}")

    img = Image.open(f"{DATA_ROOT}/{SEQ}/{SEQ}_{pic_num} 1.tga")

    plt.imshow(img)
    plt.title(f"Δaz={delta_az}, Δel={delta_el}")
    plt.axis("off")
    plt.show()

def check4():
    from dataset2 import LaserDataset
    import torch

    DATA_ROOT = "/home/nvidia/Documents/kam_ml"

    dataset = LaserDataset(
        data_root=DATA_ROOT,
        seqs=["seq2", "seq3", "seq4"],
        img_size=(320, 180)
    )

    print("Total samples:", len(dataset))

    x, y = dataset[0]

    print("Image tensor shape:", x.shape)
    print("Label tensor:", y)
    print("Label dtype:", y.dtype)

    # Range sanity
    print("Label min/max (example):", y.min().item(), y.max().item())

def check5():
    import torch
    from model import LaserNet

    model = LaserNet()

    x = torch.randn(1, 3, 320, 180)
    y = model(x)

    print("Output shape:", y.shape)
    print("Output:", y)

def inspection():
    import torch
    stats = torch.load("seq_stats.pt", map_location="cpu")

    for seq, v in stats.items():
        mean = v["mean"]
        std = v["std"]
        print(f"{seq}:")
        print(f"  mean Δaz={mean[0]:+.3f}, Δel={mean[1]:+.3f}")
        print(f"  std  Δaz={std[0]:.3f},  Δel={std[1]:.3f}")

if __name__ == '__main__':
    inspection()