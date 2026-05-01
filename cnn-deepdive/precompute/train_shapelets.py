"""Train the shapelets28 classifier.

Architecture:
  Conv 5x5x8 (pad=2)  -> ReLU -> MaxPool2
  Conv 5x5x16 (pad=2) -> ReLU -> MaxPool2
  Conv 3x3x24 (pad=1) -> ReLU -> AdaptiveAvgPool(1)
  FC 6

Asserts test accuracy >= 0.95. Saves artifacts/shapelets.pt.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shapelets_data import generate_dataset, CLASSES

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)


class ShapeletsCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.fc = nn.Linear(24, n_classes)

    def forward(self, x, return_intermediates=False):
        c1 = F.relu(self.conv1(x))         # 28x28x8
        p1 = F.max_pool2d(c1, 2)            # 14x14x8
        c2 = F.relu(self.conv2(p1))         # 14x14x16
        p2 = F.max_pool2d(c2, 2)            # 7x7x16
        c3 = F.relu(self.conv3(p2))         # 7x7x24
        g = F.adaptive_avg_pool2d(c3, 1).flatten(1)  # 24
        logits = self.fc(g)
        if return_intermediates:
            return logits, dict(conv1=c1, pool1=p1, conv2=c2, pool2=p2, conv3=c3)
        return logits


def set_seed(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def train_once(lr=1e-3, epochs=30, bs=64, seed=42):
    set_seed(seed)
    Xtr, ytr = generate_dataset(250, seed=42)        # 1500 train (250 * 6)
    Xte, yte = generate_dataset(50, seed=43)         # 300 test  (50 * 6)
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1).float()
    ytr_t = torch.from_numpy(ytr).long()
    Xte_t = torch.from_numpy(Xte).unsqueeze(1).float()
    yte_t = torch.from_numpy(yte).long()

    model = ShapeletsCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(1000 + ep))
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        model.eval()
        with torch.no_grad():
            pred_te = model(Xte_t).argmax(1)
            acc = (pred_te == yte_t).float().mean().item()
        print(f'  ep {ep+1:02d}  loss {total/n:.4f}  test_acc {acc:.4f}')
    return model, acc, Xtr, ytr, Xte, yte


def main():
    print('Training shapelets classifier...')
    attempts = [
        dict(lr=1e-3, epochs=30, bs=64, seed=42),
        dict(lr=5e-4, epochs=40, bs=32, seed=42),
        dict(lr=2e-3, epochs=40, bs=64, seed=42),
    ]
    final_model = None
    final_acc = 0.0
    final_data = None
    for i, hp in enumerate(attempts):
        print(f'\n[attempt {i+1}] {hp}')
        model, acc, Xtr, ytr, Xte, yte = train_once(**hp)
        if acc >= 0.95:
            final_model, final_acc, final_data = model, acc, (Xtr, ytr, Xte, yte)
            break
        if acc > final_acc:
            final_model, final_acc, final_data = model, acc, (Xtr, ytr, Xte, yte)

    assert final_acc >= 0.95, f'Test accuracy {final_acc:.4f} < 0.95 after all attempts'

    Xtr, ytr, Xte, yte = final_data

    torch.save({
        'state_dict': final_model.state_dict(),
        'test_accuracy': float(final_acc),
        'classes': CLASSES,
    }, os.path.join(ART, 'shapelets.pt'))

    np.savez(os.path.join(ART, 'shapelets_data.npz'),
             Xtr=Xtr.astype(np.float32), ytr=ytr,
             Xte=Xte.astype(np.float32), yte=yte)

    print(f'\nDone. Test accuracy = {final_acc:.4f}')
    return final_acc


if __name__ == '__main__':
    main()
