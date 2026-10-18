import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Lightweight DOWN block (like original Down but tiny)
# -----------------------------
class DownTiny(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.stage1 = nn.Conv2d(in_ch, embed_dim, 3, 2, 1)  # /2
        self.stage2 = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)  # /4
        self.stage3 = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)  # /8

    def forward(self, x):
        f0 = self.stage1(x)
        f1 = self.stage2(f0)
        f2 = self.stage3(f1)
        return [f0, f1, f2]


# -----------------------------
# Lightweight UP block (tiny version of FineUp)
# -----------------------------
class UpTiny(nn.Module):
    def __init__(self, embed_dim, out_ch):
        super().__init__()

        self.up2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.up1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.up0 = nn.Conv2d(embed_dim, out_ch, 3, 1, 1)

    def forward(self, coarse0, diff1, fine, coarse1):
        # 3 scales: f0 (/2), f1 (/4), f2 (/8)

        # deepest scale
        x = fine[2] + diff1[2] + coarse0[2] + coarse1[2]

        # upsample to /4
        x = F.interpolate(x, size=fine[1].shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(x + fine[1] + diff1[1] + coarse0[1] + coarse1[1])

        # upsample to /2
        x = F.interpolate(x, size=fine[0].shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(x + fine[0] + diff1[0] + coarse0[0] + coarse1[0])

        # upsample to full resolution
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up0(x)
        return x


# -----------------------------
# Simple Attention Layer
# -----------------------------
class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x.permute(0, 2, 1).view(b, c, h, w)


# -----------------------------
# FINAL MODEL (with correct upsampling)
# -----------------------------
class SwinNanoSTFM(nn.Module):
    def __init__(self, in_ch=6, out_ch=6, embed_dim=32):
        super().__init__()
        self.coarse_down = DownTiny(in_ch, embed_dim)
        self.fine_down   = DownTiny(in_ch, embed_dim)
        self.fine_up     = UpTiny(embed_dim, out_ch)

        # attention blocks
        self.block1 = SimpleAttention(embed_dim)
        self.block2 = SimpleAttention(embed_dim)

    def forward(self, c0, f0, c1):

        # Downsample (multi-scale)
        coarse0 = self.coarse_down(c0)
        coarse1 = self.coarse_down(c1)
        fine    = self.fine_down(f0)

        # attention across scales #inserted [2] for course 0 and course 1 to prevent quadratic memory blowup
        coarse0[2] = self.block1(coarse0[2])
        coarse1[2] = self.block1(coarse1[2])
        fine[2]    = self.block2(fine[2])

        # difference features
        diff1 = [coarse1[i] - coarse0[i] for i in range(3)]
        # note: fine - coarse0 already okay at same scales
        
        # diff0 is never used in UpTiny.You can safely remove this line to save memory slightly:
        #diff0 = [fine[i] - coarse0[i] for i in range(3)]

        # upsample and decode to full resolution
        out = self.fine_up(coarse0, diff1, fine, coarse1)

        return out
