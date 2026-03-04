# export_onnx.py
"""
Export a trained SimpleNet checkpoint to ONNX.

Usage:
    python export_onnx.py --model_dir ./runs/best --out model.onnx

The resulting .onnx can be:
  - Used directly with ONNX Runtime on the Pi CPU (fast, no extra toolchain)
  - Compiled to Hailo HEF using Hailo Dataflow Compiler on an x86 machine:
      hailo optimize model.onnx --hw-arch hailo8l
      hailo compile  model.onnx --hw-arch hailo8l -o model.hef
"""

import argparse
import torch
from model import SimpleNet
from dataset import N_PARAMS, IMG_SIZE_HW

DEFAULT_IMG_SIZE_HW = IMG_SIZE_HW


def export(model_dir: str, out_path: str, img_size_hw=DEFAULT_IMG_SIZE_HW):
    config_path  = f"{model_dir}/config.pt"
    weights_path = f"{model_dir}/best_model.pt"

    out_scale = 2.6
    try:
        cfg = torch.load(config_path, map_location="cpu")
        out_scale = cfg.get("out_scale", 2.6)
        print(f"config.pt: out_scale={out_scale}")
    except FileNotFoundError:
        print(f"[WARN] config.pt not found, using out_scale={out_scale}")

    model = SimpleNet(in_channels=6, n_params=N_PARAMS, out_scale=out_scale)
    ckpt  = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    H, W  = img_size_hw
    dummy = torch.zeros(1, 6, H, W)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {out_path}  (input shape [B, 6, {H}, {W}]  ->  output [B, {N_PARAMS}])")

    # Quick sanity check
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"ONNX Runtime check passed. Output shape: {out.shape}")
    except ImportError:
        print("onnxruntime not installed — skipping verification.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./runs/best")
    p.add_argument("--out",       default="model.onnx")
    args = p.parse_args()
    export(args.model_dir, args.out)