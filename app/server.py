from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, cv2, torch
from aurora.model import SiameseUNet

app = FastAPI(title="AuroraSAR-Change API")
_model = None

def get_model():
    global _model
    if _model is None:
        m = SiameseUNet()
        m.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
        m.eval()
        _model = m
    return _model

def to_gray_tensor(raw: bytes, size=64):
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError("Invalid image")
    img = cv2.resize(img, (size,size))
    t = (img/255.0).astype(np.float32)[None,None,...]
    return torch.from_numpy(t)

@app.get("/")
def root(): return {"ok": True, "info":"POST /infer"}

@app.post("/infer")
async def infer(before: UploadFile = File(...), after: UploadFile = File(...)):
    b = to_gray_tensor(await before.read())
    a = to_gray_tensor(await after.read())
    m = get_model()
    with torch.no_grad():
        prob = torch.sigmoid(m(b,a))[0,0].numpy()
    heat = (prob*255).astype(np.uint8)
    # return as list for simplicity
    return {"width": int(heat.shape[1]), "height": int(heat.shape[0]),
            "heat_uint8_flat": heat.flatten().tolist()}
