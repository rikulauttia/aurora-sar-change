import cv2, torch, numpy as np
from aurora.model import SiameseUNet

def load_gray(path, size=64):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size,size))
    t = (img/255.0).astype(np.float32)[None,None,...]
    return torch.from_numpy(t)

def infer_pair(before_path, after_path, ckpt="checkpoint.pt", size=64):
    model = SiameseUNet()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    b = load_gray(before_path, size)
    a = load_gray(after_path, size)
    with torch.no_grad():
        prob = torch.sigmoid(model(b,a))[0,0].numpy()
    heat = (prob*255).astype(np.uint8)
    return heat

if __name__ == "__main__":
    heat = infer_pair("data/pairs/demo_002_before.png","data/pairs/demo_002_after.png")
    cv2.imwrite("change_heat.png", heat)
    print("Wrote change_heat.png")
