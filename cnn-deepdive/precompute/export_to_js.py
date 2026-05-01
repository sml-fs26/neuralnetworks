"""Assemble all artifacts into ``data/datasets.js``.

Reads:
  artifacts/shapelets.pt
  artifacts/shapelets_data.npz
  artifacts/segmenter.pt
  artifacts/scene64_data.npz
  artifacts/am.npz
  artifacts/conv2_top9.npz
  artifacts/receptive_fields.json

Produces:
  ../data/datasets.js
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from train_shapelets import ShapeletsCNN
from train_segmenter import TinyUNet
from hand_filters import build as build_hand_filters
from receptive_fields import compute as compute_rf

ROUND = 3  # reduced from 4 to keep file under 4 MB budget
HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, 'artifacts')
OUT = os.path.normpath(os.path.join(HERE, '..', 'data', 'datasets.js'))


def r(x):
    """Round to ROUND decimals; return Python float (avoids numpy types)."""
    return round(float(x), ROUND)


def r_acc(x):
    """Higher-precision rounding for scalar accuracies (so 0.9998 stays
    distinguishable from 1.0)."""
    return round(float(x), 4)


def to_list(arr):
    """Recursively convert numpy array (or scalar) to nested Python lists with
    rounded floats. Integer arrays preserve int."""
    if isinstance(arr, (np.floating, float)):
        return r(arr)
    if isinstance(arr, (np.integer, int)):
        return int(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == 'f':
            return np.round(arr.astype(np.float64), ROUND).tolist()
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return [to_list(x) for x in arr]
    raise TypeError(f'Cannot convert {type(arr)}')


def js_repr(value):
    """Render a Python value as a compact JS literal. Uses json with rounding
    already applied. Floats stay as floats; integers as integers."""
    return json.dumps(value, separators=(',', ':'))


def normalize_filter_for_display(k):
    """Min-max normalize a 2D kernel to [0, 1] for direct image display."""
    k = np.asarray(k, dtype=np.float32)
    lo, hi = k.min(), k.max()
    if hi - lo < 1e-9:
        return np.full_like(k, 0.5)
    return (k - lo) / (hi - lo)


def gather_shapelets():
    print('Assembling shapelets section...')
    ckpt = torch.load(os.path.join(ART, 'shapelets.pt'), map_location='cpu')
    model = ShapeletsCNN()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    data = np.load(os.path.join(ART, 'shapelets_data.npz'))
    Xtr, ytr = data['Xtr'], data['ytr']
    Xte, yte = data['Xte'], data['yte']

    classes = list(ckpt['classes'])

    sd = ckpt['state_dict']
    conv1_w = sd['conv1.weight'].numpy()    # (8,1,5,5)
    conv1_b = sd['conv1.bias'].numpy()
    conv2_w = sd['conv2.weight'].numpy()    # (16,8,5,5)
    conv2_b = sd['conv2.bias'].numpy()
    conv3_w = sd['conv3.weight'].numpy()    # (24,16,3,3)
    conv3_b = sd['conv3.bias'].numpy()
    fc_w = sd['fc.weight'].numpy()          # (6,24)
    fc_b = sd['fc.bias'].numpy()

    # 6 sample inputs, one per class. Pick the first-encountered training image
    # of each class so it's reproducible.
    samples = []
    for cls_idx, cls_name in enumerate(classes):
        # Use a TEST image (cleaner showcase); fall back to train if none.
        idx_cands = np.where(yte == cls_idx)[0]
        if len(idx_cands) == 0:
            idx_cands = np.where(ytr == cls_idx)[0]
            X_src = Xtr
        else:
            X_src = Xte
        idx = int(idx_cands[0])
        x = X_src[idx]
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits, inter = model(x_t, return_intermediates=True)
            probs = F.softmax(logits, dim=1)
        sample = dict(
            cls=cls_name,
            input=to_list(x),
            conv1Out=to_list(inter['conv1'][0].numpy()),    # (8,28,28)
            pool1Out=to_list(inter['pool1'][0].numpy()),    # (8,14,14)
            conv2Out=to_list(inter['conv2'][0].numpy()),    # (16,14,14)
            pool2Out=to_list(inter['pool2'][0].numpy()),    # (16,7,7)
            conv3Out=to_list(inter['conv3'][0].numpy()),    # (24,7,7)
            logits=to_list(logits[0].numpy()),
            probs=to_list(probs[0].numpy()),
        )
        samples.append(sample)

    # Pre-rendered conv1 filter visualizations normalized to [0,1] for direct display.
    conv1_filt_norm = []
    for c in range(conv1_w.shape[0]):
        k = conv1_w[c, 0]  # (5,5)
        conv1_filt_norm.append(to_list(normalize_filter_for_display(k)))

    # conv2 filters collapsed to single 5x5 by mean over input channels, then normalized.
    conv2_filt_norm = []
    for c in range(conv2_w.shape[0]):
        k = conv2_w[c].mean(axis=0)  # (5,5)
        conv2_filt_norm.append(to_list(normalize_filter_for_display(k)))

    # conv2 top-9 indices (over training set, all 16 channels).
    conv2_top9 = np.load(os.path.join(ART, 'conv2_top9.npz'))['top9']  # (16, 9)

    # 40 sample train images (round to 4 dp).
    rng = np.random.RandomState(123)
    sample_train_idx = rng.choice(len(Xtr), size=40, replace=False)
    sample_train_idx.sort()
    train_images_sample = [to_list(Xtr[i]) for i in sample_train_idx]

    # Map any indices used in conv2_top9 (which index into Xtr) so the JS
    # consumer can resolve them. The brief says "indices into
    # shapelets.trainImagesSample". Re-map each top-9 index to either
    # (a) the position within sample_train_idx, or (b) include the actual
    # image. The spec is ambiguous so we provide BOTH: the dense Xtr index
    # and (since runtime probably wants the image directly) we'll include
    # the AM neurons' top9 with full indices, and supply a denser
    # trainImagesSample. To remain safe, we extend the train images sample
    # to also cover ALL conv2 top-9 indices used.
    extra = set(int(i) for row in conv2_top9 for i in row)
    base = set(int(i) for i in sample_train_idx)
    new = sorted(base | extra)
    new_idx_map = {old_idx: new_pos for new_pos, old_idx in enumerate(new)}
    train_images_full = [to_list(Xtr[i]) for i in new]
    conv2_top9_remapped = [
        [new_idx_map[int(i)] for i in row] for row in conv2_top9
    ]

    return dict(
        classes=classes,
        conv1=dict(kernels=to_list(conv1_w), biases=to_list(conv1_b)),
        conv2=dict(kernels=to_list(conv2_w), biases=to_list(conv2_b)),
        conv3=dict(kernels=to_list(conv3_w), biases=to_list(conv3_b)),
        fc=dict(weights=to_list(fc_w), biases=to_list(fc_b)),
        testAccuracy=r_acc(ckpt['test_accuracy']),
        samples=samples,
        conv1FiltersNormalized=conv1_filt_norm,
        conv2FiltersNormalized=conv2_filt_norm,
        conv2Top9=conv2_top9_remapped,
        trainImagesSample=train_images_full,
    )


def gather_scene64():
    print('Assembling scene64 section...')
    ckpt = torch.load(os.path.join(ART, 'segmenter.pt'), map_location='cpu')
    model = TinyUNet()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    data = np.load(os.path.join(ART, 'scene64_data.npz'))
    Xte, Yte = data['Xte'], data['Yte']

    # Pick 6 test scenes that together cover all 5 classes.
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
    # Top up to 6 if not yet
    for i in range(len(Xte)):
        if len(chosen) >= 6:
            break
        if i not in chosen:
            chosen.append(i)

    samples = []
    with torch.no_grad():
        for i in chosen[:6]:
            x = Xte[i]    # (3,64,64)
            y = Yte[i]    # (64,64)
            x_t = torch.from_numpy(x).unsqueeze(0).float()
            out, inter = model(x_t, return_intermediates=True)
            pred = out.argmax(1).squeeze(0).numpy()
            # Convert image to (64,64,3) order for the JS schema.
            img_hwc = np.transpose(x, (1, 2, 0))  # (64,64,3)
            sample = dict(
                input=to_list(img_hwc),
                label=to_list(y),
                pred=to_list(pred),
                enc1=to_list(inter['enc1'][0, :4].numpy()),
                enc2=to_list(inter['enc2'][0, :4].numpy()),
                enc3=to_list(inter['enc3'][0, :4].numpy()),
                dec2=to_list(inter['dec2'][0, :4].numpy()),
                dec1=to_list(inter['dec1'][0, :4].numpy()),
            )
            samples.append(sample)

    # Inline weights by layer name.
    sd = ckpt['state_dict']
    weights = {k: to_list(v.numpy()) for k, v in sd.items()}

    return dict(
        classes=list(ckpt['classes']),
        samples=samples,
        meanPixelAccuracy=r_acc(ckpt['mean_pixel_accuracy']),
        weights=weights,
    )


def gather_AM():
    print('Assembling AM section...')
    npz = np.load(os.path.join(ART, 'am.npz'))
    layers = npz['layers']
    channels = npz['channels']
    images = npz['images']    # (N, 28, 28)
    top9 = npz['top9']        # (N, 9)
    neurons = []
    for i in range(len(layers)):
        neurons.append(dict(
            layer=str(layers[i]),
            channel=int(channels[i]),
            image=to_list(images[i]),
            top9Indices=[int(x) for x in top9[i]],
        ))
    return dict(neurons=neurons)


def gather_RF():
    return compute_rf()


def gather_hand_filters():
    F_ = build_hand_filters()
    return {k: to_list(v) for k, v in F_.items()}


def emit():
    parts = []
    parts.append('window.DATA = window.DATA || {};\n')

    parts.append('window.DATA.handFilters = ')
    parts.append(js_repr(gather_hand_filters()))
    parts.append(';\n')

    shape = gather_shapelets()
    # samples have a 'cls' field internally that we want emitted as 'class'.
    # Build the JS structure carefully.
    train_images_sample = shape.pop('trainImagesSample')
    samples_js = []
    for s in shape['samples']:
        s2 = dict(s)
        s2['class'] = s2.pop('cls')
        samples_js.append(s2)
    shape['samples'] = samples_js

    parts.append('window.DATA.shapelets = ')
    parts.append(js_repr(shape))
    parts.append(';\n')

    parts.append('window.DATA.shapelets.trainImagesSample = ')
    parts.append(js_repr(train_images_sample))
    parts.append(';\n')

    parts.append('window.DATA.scene64 = ')
    parts.append(js_repr(gather_scene64()))
    parts.append(';\n')

    parts.append('window.DATA.AM = ')
    parts.append(js_repr(gather_AM()))
    parts.append(';\n')

    parts.append('window.DATA.RF = ')
    parts.append(js_repr(gather_RF()))
    parts.append(';\n')

    out = ''.join(parts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(out)
    size = os.path.getsize(OUT)
    print(f'Wrote {OUT} ({size:,} bytes)')
    return size


def main():
    size = emit()
    if size > 4 * 1024 * 1024:
        print(f'WARNING: file size {size} exceeds 4 MB budget.')
    return size


if __name__ == '__main__':
    main()
