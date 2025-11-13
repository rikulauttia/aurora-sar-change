import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tiny UNet blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)

# --- Siamese UNet-lite (before & after -> change mask) ---
class SiameseUNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        # two encoders share weights (siamese)
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = Down(base, base*2)
        self.enc3 = Down(base*2, base*4)

        # shared for both inputs
        self.enc1_b = self.enc1
        self.enc2_b = self.enc2
        self.enc3_b = self.enc3

        # fuse enc3 features (concat)
        self.bottleneck = ConvBlock(base*8, base*8)

        # decoder
        self.up1 = Up(base*8, base*4)
        self.up2 = Up(base*4, base*2)
        self.outc = nn.Conv2d(base*2, 1, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return [e1, e2, e3]

    def forward(self, before, after):
        b1, b2, b3 = self.encode(before)
        a1, a2, a3 = self.encode(after)
        fused = torch.cat([b3, a3], dim=1)
        x = self.bottleneck(fused)
        x = self.up1(x, torch.cat([b2, a2], dim=1))  # fuse mid-level
        x = self.up2(x, torch.cat([b1, a1], dim=1))  # fuse low-level
        x = self.outc(x)
        return x  # raw logits
