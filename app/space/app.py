import os
import gradio as gr
import numpy as np
import cv2
import torch

# -----------------------
# Config
# -----------------------
SIZE = 64            # expected model input side
FUSE_WEIGHT = 0.5    # 50% model, 50% abs-diff
THRESH = 0.25        # binary mask threshold for quick stat

# -----------------------
# Model (optional)
# -----------------------
model = None
ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoint.pt")
try:
    from aurora.model import SiameseUNet
    if os.path.exists(ckpt_path):
        model = SiameseUNet()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        print("[Aurora] Loaded checkpoint:", ckpt_path)
    else:
        print("[Aurora] checkpoint.pt not found — running abs-diff baseline only.")
except Exception as e:
    print("[Aurora] Model import/load failed, using abs-diff baseline only. Reason:", e)
    model = None

# -----------------------
# Utils
# -----------------------
def to_gray_float(img_pil, size=SIZE):
    """PIL -> grayscale np.float32 in [0,1], resized consistently."""
    g = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_NEAREST)
    return (g.astype(np.float32) / 255.0)

def robust_norm(x, p_lo=2, p_hi=98, eps=1e-6):
    lo, hi = np.percentile(x, [p_lo, p_hi])
    den = max(hi - lo, eps)
    y = np.clip((x - lo) / den, 0.0, 1.0)
    return y

def to_overlay(base_gray_01, mask_01):
    """Color heatmap over the AFTER image (keeps the scene visible)."""
    heat = (mask_01 * 255).astype(np.uint8)
    heat_rgb = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base_rgb = cv2.cvtColor((base_gray_01 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(base_rgb, 0.6, heat_rgb, 0.4, 0.0)
    return overlay

# -----------------------
# Inference
# -----------------------
def run(before_img, after_img):
    if before_img is None or after_img is None:
        return None, "Please upload both images."

    # 1) Preprocess
    b = to_gray_float(before_img)
    a = to_gray_float(after_img)

    # 2) Abs-diff baseline (guarantees correct toy results)
    absdiff = np.abs(a - b)
    absdiff_norm = robust_norm(absdiff)

    # 3) Model mask (optional)
    if model is not None:
        bt = torch.from_numpy(b)[None, None, ...]  # (1,1,H,W)
        at = torch.from_numpy(a)[None, None, ...]
        with torch.no_grad():
            logits = model(bt, at)          # (1,1,H,W)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        # 4) Fuse model + abs-diff for stability
        mask = FUSE_WEIGHT * prob + (1.0 - FUSE_WEIGHT) * absdiff_norm
    else:
        mask = absdiff_norm

    # 5) Stats + overlay
    mask = np.clip(mask, 0.0, 1.0)
    mean_score = float(mask.mean())
    binary_mean = float((mask > THRESH).mean())  # fraction of pixels above threshold
    overlay = to_overlay(a, mask)

    msg = (
        f"Mean change score: **{mean_score:.4f}**  \n"
        f"Pixels above {THRESH:.2f}: **{binary_mean*100:.2f}%**  \n"
        f"Fusion: {'model+absdiff' if model is not None else 'absdiff-only'}"
    )
    return overlay, msg

# -----------------------
# UI
# -----------------------
title = "# AuroraSAR-Change — Explainable SAR change detection"
desc = (
    "Upload two grayscale (or RGB) images: **Before** and **After**. "
    "The app highlights **where** the scene changed. "
    "For this demo, try the toy pairs in `data/pairs/`:\n\n"
    "- `demo_001_*` → **no change** (overlay should be near-empty)\n"
    "- `demo_002_*` → **new white square** (overlay shows a bright square)\n"
)

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(desc)

    with gr.Row():
        before = gr.Image(type="pil", label="Before (SAR)")
        after  = gr.Image(type="pil", label="After (SAR)")

    with gr.Row():
        out_img = gr.Image(type="numpy", label="Change heat overlay")
    out_txt = gr.Markdown()

    btn = gr.Button("Run", variant="primary")
    btn.click(run, inputs=[before, after], outputs=[out_img, out_txt])

    gr.Examples(
        examples=[
            ["data/pairs/demo_001_before.png", "data/pairs/demo_001_after.png"],
            ["data/pairs/demo_002_before.png", "data/pairs/demo_002_after.png"],
        ],
        inputs=[before, after],
        label="Quick examples (toy PNGs)",
    )

if __name__ == "__main__":
    demo.launch()