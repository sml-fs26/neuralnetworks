"""B5 — Training traces for scene 13.

Re-run the with-skip TinyUNet training (seed 43, identical recipe) and
snapshot every K=20 optimizer steps:
  - running training loss (avg over the last K steps)
  - test pixel accuracy on the full 100-sample test set
  - argmax prediction on one fixed test sample (index 0)
  - Frobenius norm of up1.weight and up2.weight

600 train samples / 16 bs ≈ 38 steps per epoch × 30 epochs ≈ 1140 steps.
At K=20 that's ≈57 frames; we keep them all (not too large).

Outputs: artifacts/training_traces.npz with arrays:
  steps         : (F,)  int
  loss          : (F,)  float32
  pix_acc       : (F,)  float32
  pred_frames   : (F, 64, 64)  int64 (argmax)
  up1_norm      : (F,)  float32
  up2_norm      : (F,)  float32
  fixed_sample_input : (3, 64, 64)  float32
  fixed_sample_label : (64, 64)     int64
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene64_data import generate_dataset, CLASSES
from train_segmenter import TinyUNet, set_seed

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)

K = 20  # snapshot every K steps


def evaluate(model, Xte_t, Yte_t, bs=16):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte_t), bs):
            p = model(Xte_t[i:i + bs]).argmax(1)
            preds.append(p)
        preds = torch.cat(preds, 0)
        acc = (preds == Yte_t).float().mean().item()
    return acc


def main():
    set_seed(43)
    print('Generating data and starting traced training...')
    Xtr, Ytr = generate_dataset(600, seed=43)
    Xte, Yte = generate_dataset(100, seed=44)
    Xtr_t = torch.from_numpy(Xtr).float()
    Ytr_t = torch.from_numpy(Ytr).long()
    Xte_t = torch.from_numpy(Xte).float()
    Yte_t = torch.from_numpy(Yte).long()

    fixed_input = torch.from_numpy(Xte[0:1]).float()
    fixed_label = Yte[0]

    model = TinyUNet()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    bs = 16
    epochs = 30
    n = len(Xtr_t)

    steps_log = []
    loss_log = []
    acc_log = []
    pred_log = []
    up1_log = []
    up2_log = []

    # Snapshot at step 0 (before any training).
    with torch.no_grad():
        p0 = model(fixed_input).argmax(1).squeeze(0).numpy()
    acc0 = evaluate(model, Xte_t, Yte_t, bs=bs)
    steps_log.append(0)
    loss_log.append(float('nan'))
    acc_log.append(acc0)
    pred_log.append(p0)
    up1_log.append(float(model.up1.weight.norm().item()))
    up2_log.append(float(model.up2.weight.norm().item()))
    print(f'  step 0  loss   nan  pix_acc {acc0:.4f}  '
          f'up1 {up1_log[-1]:.3f}  up2 {up2_log[-1]:.3f}')

    step = 0
    rolling = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(2000 + ep))
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            step += 1
            rolling.append(loss.item())
            if step % K == 0:
                avg_loss = float(np.mean(rolling[-K:]))
                acc = evaluate(model, Xte_t, Yte_t, bs=bs)
                with torch.no_grad():
                    p = model(fixed_input).argmax(1).squeeze(0).numpy()
                steps_log.append(step)
                loss_log.append(avg_loss)
                acc_log.append(acc)
                pred_log.append(p)
                up1_log.append(float(model.up1.weight.norm().item()))
                up2_log.append(float(model.up2.weight.norm().item()))
                model.train()
                if step % 100 == 0 or step == K:
                    print(f'  step {step:4d}  loss {avg_loss:.4f}  '
                          f'pix_acc {acc:.4f}  up1 {up1_log[-1]:.3f}  '
                          f'up2 {up2_log[-1]:.3f}')

    # Final snapshot if we don't have one at the very end.
    if step % K != 0:
        avg_loss = float(np.mean(rolling[-(step % K):]))
        acc = evaluate(model, Xte_t, Yte_t, bs=bs)
        with torch.no_grad():
            p = model(fixed_input).argmax(1).squeeze(0).numpy()
        steps_log.append(step)
        loss_log.append(avg_loss)
        acc_log.append(acc)
        pred_log.append(p)
        up1_log.append(float(model.up1.weight.norm().item()))
        up2_log.append(float(model.up2.weight.norm().item()))

    print(f'Total snapshots: {len(steps_log)}')
    out_path = os.path.join(ART, 'training_traces.npz')
    np.savez(out_path,
             steps=np.asarray(steps_log, dtype=np.int64),
             loss=np.asarray(loss_log, dtype=np.float32),
             pix_acc=np.asarray(acc_log, dtype=np.float32),
             pred_frames=np.stack(pred_log, axis=0).astype(np.int64),
             up1_norm=np.asarray(up1_log, dtype=np.float32),
             up2_norm=np.asarray(up2_log, dtype=np.float32),
             fixed_sample_input=Xte[0].astype(np.float32),
             fixed_sample_label=fixed_label.astype(np.int64))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
