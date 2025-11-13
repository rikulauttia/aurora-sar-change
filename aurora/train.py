import os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from aurora.model import SiameseUNet
from aurora.dataset import PairDataset

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = (pred + target - pred*target).sum()
    return (inter + eps) / (union + eps)

def train():
    ds = PairDataset(root="data/pairs", size=64)
    if len(ds) < 2:
        raise RuntimeError("Add at least 2 pairs in data/pairs.")
    n_val = max(1, len(ds)//3)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    vl = DataLoader(val_ds, batch_size=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_iou, best_path = 0.0, "checkpoint.pt"
    for epoch in range(10):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/10")
        for b,a,m in pbar:
            b,a,m = b.to(device), a.to(device), m.to(device)
            logits = model(b,a)
            loss = F.binary_cross_entropy_with_logits(logits, m)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": float(loss.item())})

        # val
        model.eval()
        ious = []
        with torch.no_grad():
            for b,a,m in vl:
                b,a,m = b.to(device), a.to(device), m.to(device)
                logits = model(b,a)
                prob = torch.sigmoid(logits)
                ious.append(float(iou_score(prob, m).item()))
        miou = sum(ious)/len(ious)
        print(f"Val mIoU: {miou:.3f}")
        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), best_path)
            print(f"Saved {best_path} (mIoU={best_iou:.3f})")

if __name__ == "__main__":
    train()
