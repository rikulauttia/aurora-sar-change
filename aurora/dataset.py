import os, glob, cv2, numpy as np, torch
from torch.utils.data import Dataset

def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

class PairDataset(Dataset):
    def __init__(self, root="data/pairs", size=64):
        self.size = size
        self.before = sorted(glob.glob(os.path.join(root, "*_before.png")))
        self.after  = [p.replace("_before.png","_after.png") for p in self.before]
        self.mask   = [p.replace("_before.png","_mask.png")  for p in self.before]

    def __len__(self): return len(self.before)

    def __getitem__(self, i):
        b = _read_gray(self.before[i])
        a = _read_gray(self.after[i])
        m = _read_gray(self.mask[i])

        # ensure HxW == size
        b = cv2.resize(b, (self.size, self.size))
        a = cv2.resize(a, (self.size, self.size))
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # normalize to [0,1]
        b = (b / 255.0).astype(np.float32)[None, ...]
        a = (a / 255.0).astype(np.float32)[None, ...]
        m = (m > 127).astype(np.float32)[None, ...]  # binary

        return torch.from_numpy(b), torch.from_numpy(a), torch.from_numpy(m)
