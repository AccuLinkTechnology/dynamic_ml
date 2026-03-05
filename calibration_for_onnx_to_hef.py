import numpy as np, os, glob
from PIL import Image
import torchvision.transforms as T
import torch

#calibrates conversion from float32 onnx to int8 data based on realistic data.
#calibration data: at some point get this from inference on real frames versus generated data.


transform = T.Compose([
    T.Resize((180, 320), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

os.makedirs("calib_data", exist_ok=True)
seqs = sorted(glob.glob("train3/seq*"))[:5]  # 5 sequences is plenty
count = 0
for seq in seqs:
    ref = transform(Image.open(f"{seq}/{os.path.basename(seq)}_start.jpg").convert("RGB"))
    for img_path in sorted(glob.glob(f"{seq}/seq*.jpg")):
        if "_start" in img_path:
            continue
        cur = transform(Image.open(img_path).convert("RGB"))
        x_in = torch.cat([cur, cur - ref], dim=0).unsqueeze(0)
        np.save(f"calib_data/{count:04d}.npy", x_in.numpy())
        count += 1
        if count >= 200:
            break
    if count >= 200:
        break
print(f"Saved {count} calibration frames")