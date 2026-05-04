"""B2 — Upsampling demos for scene 6.

Three small input maps × three upsampling methods. The transposed-conv branch
also varies over three hand-picked filters so the viewer sees that the
transposed-conv answer depends on the *filter*, while nearest and bilinear
are filter-free.

Inputs (each 4×4, single channel):
  - bright_cell : a single bright pixel near the center.
  - small_cross : a 3-pixel "+" shape.
  - diagonal    : three pixels along a diagonal.

Methods (all upscale 4×4 -> 8×8):
  - nearest     : torch interpolate, mode='nearest'.
  - bilinear    : torch interpolate, mode='bilinear', align_corners=False.
  - tconv       : nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2,
                  padding=1, output_padding=1) with weight set to a fixed 3×3
                  filter. Three filters: gaussian, plus, edge.

Emits artifacts/upsample_demos.json:
  {
    inputs: { name: [[...4×4]] , ... },
    outputs: {
       name: {
         nearest: [[...8×8]],
         bilinear: [[...8×8]],
         tconv: { gaussian: [[...8×8]], plus: ..., edge: ... }
       },
       ...
    },
    filters: { gaussian: [[...3×3]], plus: ..., edge: ... }
  }
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)
ROUND = 4


def r(arr):
    return np.round(np.asarray(arr, dtype=np.float64), ROUND).tolist()


def make_inputs():
    inputs = {}
    a = np.zeros((4, 4), dtype=np.float32)
    a[1, 2] = 1.0
    inputs['bright_cell'] = a

    b = np.zeros((4, 4), dtype=np.float32)
    b[1, 2] = 1.0
    b[2, 1] = 1.0
    b[2, 2] = 1.0
    b[2, 3] = 1.0
    b[3, 2] = 1.0
    inputs['small_cross'] = b

    c = np.zeros((4, 4), dtype=np.float32)
    c[0, 0] = 1.0
    c[1, 1] = 1.0
    c[2, 2] = 1.0
    c[3, 3] = 1.0
    inputs['diagonal'] = c
    return inputs


def make_filters():
    # Each is 3×3.
    gaussian = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.float32) / 16.0
    plus = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.float32) / 5.0
    edge = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=np.float32) / 8.0
    return dict(gaussian=gaussian, plus=plus, edge=edge)


def nearest_2x(x):
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y = F.interpolate(t, scale_factor=2, mode='nearest')
    return y.squeeze().numpy()


def bilinear_2x(x):
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
    return y.squeeze().numpy()


def tconv_2x(x, k):
    """Stride-2 transposed conv with the given 3×3 kernel.

    The exact output size depends on padding/output_padding; we want 8×8 from
    4×4 with a 3×3 kernel and stride 2 — set padding=1, output_padding=1.
    """
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    layer = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2,
                                padding=1, output_padding=1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(k).float().unsqueeze(0).unsqueeze(0))
        y = layer(t)
    return y.squeeze().numpy()


def main():
    inputs = make_inputs()
    filters = make_filters()
    outputs = {}
    for name, x in inputs.items():
        nn_out = nearest_2x(x)
        bl_out = bilinear_2x(x)
        tc_outs = {fname: tconv_2x(x, fk) for fname, fk in filters.items()}
        outputs[name] = dict(
            nearest=r(nn_out),
            bilinear=r(bl_out),
            tconv={fn: r(v) for fn, v in tc_outs.items()},
        )

    bundle = dict(
        inputs={k: r(v) for k, v in inputs.items()},
        outputs=outputs,
        filters={k: r(v) for k, v in filters.items()},
    )
    out_path = os.path.join(ART, 'upsample_demos.json')
    with open(out_path, 'w') as f:
        json.dump(bundle, f, separators=(',', ':'))
    print(f'Wrote {out_path}')
    # quick sanity print
    for k in inputs:
        print(f'  {k}: nearest sum {np.sum(outputs[k]["nearest"]):.3f}, '
              f'bilinear sum {np.sum(outputs[k]["bilinear"]):.3f}')


if __name__ == '__main__':
    main()
