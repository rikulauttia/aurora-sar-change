import os
import numpy as np
import cv2

OUT = "data/pairs"
SIZE = 64  # 64x64 grayscale

os.makedirs(OUT, exist_ok=True)

def write_png(path, arr):
    cv2.imwrite(path, arr)

# --- demo_001: NO CHANGE ---
img = np.zeros((SIZE, SIZE), np.uint8)
# draw a simple square in the middle
cv2.rectangle(img, (20,20), (44,44), 180, thickness=-1)

write_png(f"{OUT}/demo_001_before.png", img)
write_png(f"{OUT}/demo_001_after.png",  img.copy())      # same image
write_png(f"{OUT}/demo_001_mask.png",   np.zeros_like(img))  # all zeros mask

# --- demo_002: HAS CHANGE ---
bg = np.zeros((SIZE, SIZE), np.uint8)

before = bg.copy()
after  = bg.copy()
# add a bright square only to "after"
cv2.rectangle(after, (28,28), (50,50), 255, thickness=-1)

mask = np.zeros((SIZE, SIZE), np.uint8)
cv2.rectangle(mask, (28,28), (50,50), 255, thickness=-1)  # binary mask of the change

write_png(f"{OUT}/demo_002_before.png", before)
write_png(f"{OUT}/demo_002_after.png",  after)
write_png(f"{OUT}/demo_002_mask.png",   mask)

print("Wrote 6 PNGs to", OUT)