"""B6 — Confusion matrix and failure picks.

Loads the trained TinyUNet from B1, runs on the 100-sample test set, and:

  - Computes the 5×5 confusion matrix, row-normalized (rows = true class).
  - Identifies the 3 BEST and 5 WORST samples by per-sample pixel accuracy.
  - For each picked sample, records auto-generated annotations driven by
    pixel counts of each class in the label and the prediction.

Outputs:
  artifacts/confusion_matrix.json  { matrix: [[..]], classes: [..] }
  artifacts/failure_picks.json     { best: [...], worst: [...] }
"""

import json
import os
import numpy as np
import torch

from scene64_data import generate_dataset, CLASSES
from train_segmenter import TinyUNet

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)


def annotate(label, pred, classes):
    """Heuristic annotation driven by data only.

    Reports:
      - The class names actually present in the label.
      - Per-class accuracy on this sample.
      - A short data-driven note: 'tiny sun (k pixels)', 'no sun visible',
        'tree-grass confusion (n pixels)', etc.
    """
    H, W = label.shape
    total = H * W
    present = []
    notes = []
    per_class_acc = {}
    counts = {c: int((label == i).sum()) for i, c in enumerate(classes)}
    for i, c in enumerate(classes):
        if counts[c] > 0:
            present.append(c)
            mask = (label == i)
            acc = float((pred[mask] == i).sum()) / float(mask.sum())
            per_class_acc[c] = round(acc, 3)

    # Class-size driven notes.
    if 'sun' in present and counts['sun'] <= 30:
        notes.append(f'small sun ({counts["sun"]} px)')
    if 'sun' not in present:
        notes.append('no sun')
    if 'person' in present and per_class_acc.get('person', 1.0) < 0.7:
        notes.append('person partly missed')
    if 'tree' in present and per_class_acc.get('tree', 1.0) < 0.7:
        notes.append('tree partly missed')
    # Class-confusion notes: count where label==X but pred==Y.
    confusions = []
    for i, ci in enumerate(classes):
        if counts[ci] == 0:
            continue
        for j, cj in enumerate(classes):
            if i == j:
                continue
            n = int(((label == i) & (pred == j)).sum())
            if n >= 30:
                confusions.append((n, ci, cj))
    confusions.sort(reverse=True)
    for n, ci, cj in confusions[:2]:
        notes.append(f'{ci}->{cj} ({n} px)')

    return dict(
        classes_present=present,
        class_pixel_counts=counts,
        per_class_accuracy=per_class_acc,
        notes=notes,
    )


def main():
    print('Loading segmenter and test data...')
    ckpt = torch.load(os.path.join(ART, 'segmenter.pt'), map_location='cpu')
    model = TinyUNet()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Reproduce the exact test split.
    _, _ = generate_dataset(600, seed=43)  # consume train RNG (not used)
    Xte, Yte = generate_dataset(100, seed=44)

    Xte_t = torch.from_numpy(Xte).float()
    Yte_t = torch.from_numpy(Yte).long()

    bs = 16
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte_t), bs):
            preds.append(model(Xte_t[i:i + bs]).argmax(1))
    preds = torch.cat(preds, 0).numpy()
    labels = Yte_t.numpy()

    # Per-sample accuracy.
    per_sample = (preds == labels).astype(np.float32).reshape(len(preds), -1).mean(axis=1)
    order = np.argsort(per_sample)  # ascending
    worst_idx = order[:5].tolist()
    best_idx = order[-3:][::-1].tolist()

    # Confusion matrix (5x5, row-normalized).
    K = 5
    cm = np.zeros((K, K), dtype=np.float64)
    for t in range(K):
        mask = (labels == t)
        if not mask.any():
            continue
        for p in range(K):
            cm[t, p] = float(((preds == p) & mask).sum())
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    cm_payload = dict(
        classes=CLASSES,
        matrix=np.round(cm_norm, 4).tolist(),
        raw_counts=cm.astype(np.int64).tolist(),
    )
    with open(os.path.join(ART, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_payload, f, indent=2)
    print('Wrote confusion_matrix.json')
    print('Confusion matrix (row-normalized):')
    for i, c in enumerate(CLASSES):
        row = ' '.join(f'{v:.3f}' for v in cm_norm[i])
        print(f'  {c:>7s} -> {row}')

    def pack(idx_list):
        out = []
        for i in idx_list:
            out.append(dict(
                index=int(i),
                accuracy=round(float(per_sample[i]), 4),
                annotation=annotate(labels[i], preds[i], CLASSES),
            ))
        return out

    picks = dict(
        best=pack(best_idx),
        worst=pack(worst_idx),
    )
    with open(os.path.join(ART, 'failure_picks.json'), 'w') as f:
        json.dump(picks, f, indent=2)
    print('Wrote failure_picks.json')
    print('Worst samples:')
    for p in picks['worst']:
        print(f'  idx={p["index"]:3d}  acc={p["accuracy"]:.4f}  '
              f'notes={p["annotation"]["notes"]}')
    print('Best samples:')
    for p in picks['best']:
        print(f'  idx={p["index"]:3d}  acc={p["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
