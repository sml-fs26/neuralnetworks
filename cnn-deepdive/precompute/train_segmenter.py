"""Train a tiny U-Net on scene64.

Encoder: blocks of (Conv-Conv-Pool) at widths [16, 32, 64].
Decoder: (Upsample -> concat skip -> Conv-Conv) at widths [32, 16].
Final 1x1 conv -> 5 channels.

Asserts mean test pixel accuracy >= 0.90.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene64_data import generate_dataset, CLASSES

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)   # concat enc2 (32) + up2 (32) = 64
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)   # concat enc1 (16) + up1 (16) = 32
        self.out = nn.Conv2d(16, n_classes, 1)

    def forward(self, x, return_intermediates=False):
        e1 = self.enc1(x)            # 64x64x16
        p1 = F.max_pool2d(e1, 2)     # 32x32x16
        e2 = self.enc2(p1)           # 32x32x32
        p2 = F.max_pool2d(e2, 2)     # 16x16x32
        e3 = self.enc3(p2)           # 16x16x64
        u2 = self.up2(e3)            # 32x32x32
        d2 = self.dec2(torch.cat([u2, e2], 1))  # 32x32x32
        u1 = self.up1(d2)            # 64x64x16
        d1 = self.dec1(torch.cat([u1, e1], 1))  # 64x64x16
        out = self.out(d1)           # 64x64x5
        if return_intermediates:
            return out, dict(enc1=e1, enc2=e2, enc3=e3, dec2=d2, dec1=d1)
        return out


def set_seed(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def main():
    set_seed(43)
    print('Generating segmentation data...')
    Xtr, Ytr = generate_dataset(600, seed=43)
    Xte, Yte = generate_dataset(100, seed=44)
    Xtr_t = torch.from_numpy(Xtr).float()
    Ytr_t = torch.from_numpy(Ytr).long()
    Xte_t = torch.from_numpy(Xte).float()
    Yte_t = torch.from_numpy(Yte).long()

    model = TinyUNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    bs = 16
    epochs = 30
    n = len(Xtr_t)
    final_acc = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(2000 + ep))
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xte_t), bs):
                p = model(Xte_t[i:i + bs]).argmax(1)
                preds.append(p)
            preds = torch.cat(preds, 0)
            acc = (preds == Yte_t).float().mean().item()
        final_acc = acc
        print(f'  ep {ep+1:02d}  loss {total/n:.4f}  pix_acc {acc:.4f}')

    assert final_acc >= 0.90, f'Mean pixel accuracy {final_acc:.4f} < 0.90'

    torch.save({
        'state_dict': model.state_dict(),
        'mean_pixel_accuracy': float(final_acc),
        'classes': CLASSES,
    }, os.path.join(ART, 'segmenter.pt'))

    np.savez(os.path.join(ART, 'scene64_data.npz'),
             Xtr=Xtr.astype(np.float32), Ytr=Ytr,
             Xte=Xte.astype(np.float32), Yte=Yte)

    print(f'\nDone. Mean pixel accuracy = {final_acc:.4f}')


if __name__ == '__main__':
    main()
