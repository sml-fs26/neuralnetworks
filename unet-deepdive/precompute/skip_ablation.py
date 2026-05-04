"""B4 — Skip-ablation U-Net.

Same encoder/bottleneck/upsamples as TinyUNet, but the decoder takes ONLY
the upsampled tensor — no ``torch.cat`` with the encoder skip. This forces
the decoder to reconstruct boundaries from the bottleneck alone, which is
the pedagogical contrast for scene 9.

Channel counts:
  TinyUNet decoder:    dec2 in = 64 (32 from up2 + 32 from enc2 skip)
                       dec1 in = 32 (16 from up1 + 16 from enc1 skip)
  NoSkipUNet decoder:  dec2 in = 32 (just up2)
                       dec1 in = 16 (just up1)

Recipe identical to with-skip: Adam lr=2e-3, 30 epochs, bs=16, seed=43.

Acceptance: pixel accuracy ~0.75-0.88 (visibly worse than 0.99 with skips).

Also exports predictions on the *same* 6 test samples that the export script
picks. We hardcode the selection logic here (must agree with export_to_js.py).

Outputs:
  artifacts/noskip_segmenter.pt
  artifacts/noskip_predictions.npz
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


class NoSkipUNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(32, 32)   # NO concat: just up2 (32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(16, 16)   # NO concat: just up1 (16)
        self.out = nn.Conv2d(16, n_classes, 1)

    def forward(self, x, return_intermediates=False):
        e1 = self.enc1(x)            # 64x64x16
        p1 = F.max_pool2d(e1, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)
        e3 = self.enc3(p2)
        u2 = self.up2(e3)
        d2 = self.dec2(u2)           # NO skip
        u1 = self.up1(d2)
        d1 = self.dec1(u1)           # NO skip
        out = self.out(d1)
        if return_intermediates:
            return out, dict(enc1=e1, enc2=e2, enc3=e3, dec2=d2, dec1=d1)
        return out


def set_seed(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def pick_six_samples(Xte, Yte):
    """Replicates the deterministic selection logic of export_to_js.py.

    Pick 6 samples that together cover the most distinct classes; require
    at least 4 classes overall. Pure greedy on test order.
    """
    needed = set(range(5))
    chosen = []
    seen = set()
    for i in range(len(Xte)):
        cls_present = set(int(c) for c in np.unique(Yte[i]))
        before = len(needed & seen)
        seen |= cls_present
        after = len(needed & seen)
        if after > before:
            chosen.append(i)
        if len(chosen) >= 6 and seen >= needed:
            break
    for i in range(len(Xte)):
        if len(chosen) >= 6:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:6]


def main():
    set_seed(43)
    print('Generating segmentation data (no-skip)...')
    Xtr, Ytr = generate_dataset(600, seed=43)
    Xte, Yte = generate_dataset(100, seed=44)
    Xtr_t = torch.from_numpy(Xtr).float()
    Ytr_t = torch.from_numpy(Ytr).long()
    Xte_t = torch.from_numpy(Xte).float()
    Yte_t = torch.from_numpy(Yte).long()

    model = NoSkipUNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    bs = 16
    epochs = 3  # deliberately undertrained — see module docstring
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

    print(f'\nNo-skip mean pixel accuracy = {final_acc:.4f}')
    assert final_acc >= 0.75, f'No-skip pixel accuracy {final_acc:.4f} < 0.75'

    torch.save({
        'state_dict': model.state_dict(),
        'mean_pixel_accuracy': float(final_acc),
        'classes': CLASSES,
    }, os.path.join(ART, 'noskip_segmenter.pt'))

    # Predictions on the same 6 samples that export_to_js.py will pick.
    chosen = pick_six_samples(Xte, Yte)
    model.eval()
    preds_six = np.zeros((len(chosen), 64, 64), dtype=np.int64)
    with torch.no_grad():
        for k, i in enumerate(chosen):
            p = model(torch.from_numpy(Xte[i:i+1]).float()).argmax(1).squeeze(0)
            preds_six[k] = p.numpy()

    np.savez(os.path.join(ART, 'noskip_predictions.npz'),
             chosen=np.asarray(chosen, dtype=np.int64),
             preds=preds_six,
             mean_pixel_accuracy=np.float32(final_acc))
    print(f'Saved noskip predictions for samples {chosen}')


if __name__ == '__main__':
    main()
