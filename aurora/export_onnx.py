import torch
from aurora.model import SiameseUNet
import onnx, onnxruntime as ort
import numpy as np

def export(size=64, out="aurora_sar_change.onnx"):
    model = SiameseUNet()
    model.load_state_dict(torch.load("checkpoint.pt", map_location="cpu"))
    model.eval()

    b = torch.randn(1,1,size,size)
    a = torch.randn(1,1,size,size)

    torch.onnx.export(
        model, (b,a), out,
        input_names=["before","after"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"before": {0:"N"}, "after": {0:"N"}, "logits": {0:"N"}}
    )
    print("Exported", out)

    # runtime sanity
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    out_logits = sess.run(["logits"], {"before": b.numpy(), "after": a.numpy()})[0]
    print("ONNX OK, output shape:", out_logits.shape)

if __name__ == "__main__":
    export()
