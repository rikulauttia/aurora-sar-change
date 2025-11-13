import torch, numpy as np
from torch.utils.data import DataLoader
from aurora.model import SiameseUNet
from aurora.dataset import PairDataset

def precision_recall_f1(tp, fp, fn, eps=1e-7):
    p = tp/(tp+fp+eps); r = tp/(tp+fn+eps)
    f1 = 2*p*r/(p+r+eps)
    return p,r,f1

def main():
    ds = PairDataset("data/pairs", size=64)
    dl = DataLoader(ds, batch_size=1)
    model = SiameseUNet()
    model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
    model.eval()

    tp=fp=fn=0
    with torch.no_grad():
        for b,a,m in dl:
            logits = model(b, a)
            pred = (torch.sigmoid(logits) > 0.5).float()
            tp += (pred*m).sum().item()
            fp += (pred*(1-m)).sum().item()
            fn += ((1-pred)*m).sum().item()
    p,r,f1 = precision_recall_f1(tp,fp,fn)
    print(f"Precision: {p:.3f} Recall: {r:.3f} F1: {f1:.3f}")

if __name__ == "__main__":
    main()
