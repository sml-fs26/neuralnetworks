/* Shared CNN math + receptive-field utilities.

   All tensors are nested JS arrays of plain numbers (not typed arrays).
   Single-channel arrays are 2D: [H][W].
   Multi-channel feature stacks are 3D: [C][H][W].
   Kernel stacks are 4D: [outC][inC][kH][kW]. */
(function () {

  function zeros2D(h, w) {
    const a = new Array(h);
    for (let i = 0; i < h; i++) {
      const row = new Array(w);
      for (let j = 0; j < w; j++) row[j] = 0;
      a[i] = row;
    }
    return a;
  }

  function zeros3D(c, h, w) {
    const a = new Array(c);
    for (let i = 0; i < c; i++) a[i] = zeros2D(h, w);
    return a;
  }

  /* Single-channel 2D cross-correlation (what frameworks call "conv").
     `padding=0` means no padding; output size = input - kernel + 1.
     `padding=k` zero-pads by k on each side. Stride is always 1. */
  function conv2d(input, kernel, padding) {
    padding = padding || 0;
    const ih = input.length, iw = input[0].length;
    const kh = kernel.length, kw = kernel[0].length;
    const oh = ih + 2 * padding - kh + 1;
    const ow = iw + 2 * padding - kw + 1;
    const out = zeros2D(oh, ow);
    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        let s = 0;
        for (let ki = 0; ki < kh; ki++) {
          for (let kj = 0; kj < kw; kj++) {
            const ii = i + ki - padding;
            const jj = j + kj - padding;
            if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
              s += input[ii][jj] * kernel[ki][kj];
            }
          }
        }
        out[i][j] = s;
      }
    }
    return out;
  }

  /* Multi-channel conv layer.
     inputChannels: [inC][H][W]
     kernels:       [outC][inC][kH][kW]
     biases:        [outC]
     Returns:       [outC][outH][outW] */
  function multiConv2d(inputChannels, kernels, biases, padding) {
    padding = padding || 0;
    const outC = kernels.length;
    const inC = inputChannels.length;
    const ih = inputChannels[0].length;
    const iw = inputChannels[0][0].length;
    const kh = kernels[0][0].length;
    const kw = kernels[0][0][0].length;
    const oh = ih + 2 * padding - kh + 1;
    const ow = iw + 2 * padding - kw + 1;
    const out = zeros3D(outC, oh, ow);
    for (let oc = 0; oc < outC; oc++) {
      for (let i = 0; i < oh; i++) {
        for (let j = 0; j < ow; j++) {
          let s = biases ? biases[oc] : 0;
          for (let ic = 0; ic < inC; ic++) {
            for (let ki = 0; ki < kh; ki++) {
              for (let kj = 0; kj < kw; kj++) {
                const ii = i + ki - padding;
                const jj = j + kj - padding;
                if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
                  s += inputChannels[ic][ii][jj] * kernels[oc][ic][ki][kj];
                }
              }
            }
          }
          out[oc][i][j] = s;
        }
      }
    }
    return out;
  }

  function relu2D(x) {
    const h = x.length, w = x[0].length;
    const out = zeros2D(h, w);
    for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) out[i][j] = x[i][j] > 0 ? x[i][j] : 0;
    return out;
  }

  function relu3D(x) {
    return x.map(relu2D);
  }

  /* Non-overlapping max pool with given window size (and stride = window). */
  function maxpool2d(x, size) {
    const h = x.length, w = x[0].length;
    const oh = Math.floor(h / size);
    const ow = Math.floor(w / size);
    const out = zeros2D(oh, ow);
    for (let i = 0; i < oh; i++) {
      for (let j = 0; j < ow; j++) {
        let m = -Infinity;
        for (let di = 0; di < size; di++) {
          for (let dj = 0; dj < size; dj++) {
            const v = x[i * size + di][j * size + dj];
            if (v > m) m = v;
          }
        }
        out[i][j] = m;
      }
    }
    return out;
  }

  function maxpool3d(x, size) {
    return x.map(c => maxpool2d(c, size));
  }

  /* Patch extraction at (top, left) of size kH×kW, with implicit zero padding
     where requested. Returns a 2D array. */
  function extractPatch(input, top, left, kH, kW) {
    const ih = input.length, iw = input[0].length;
    const out = zeros2D(kH, kW);
    for (let i = 0; i < kH; i++) {
      for (let j = 0; j < kW; j++) {
        const ii = top + i, jj = left + j;
        if (ii >= 0 && ii < ih && jj >= 0 && jj < iw) {
          out[i][j] = input[ii][jj];
        }
      }
    }
    return out;
  }

  /* Elementwise product + sum. Returns { product: 2D, sum: scalar }.
     Used by scene 2 to animate the dot product step by step. */
  function dotProductBreakdown(patch, kernel) {
    const h = patch.length, w = patch[0].length;
    const product = zeros2D(h, w);
    let sum = 0;
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const v = patch[i][j] * kernel[i][j];
        product[i][j] = v;
        sum += v;
      }
    }
    return { product, sum };
  }

  /* Stats on a 2D tensor. */
  function range2D(x) {
    let lo = Infinity, hi = -Infinity;
    for (const row of x) for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) lo = 0;
    if (!isFinite(hi)) hi = 0;
    return { lo, hi };
  }

  /* Receptive-field rectangle in input coords for a single neuron at (row, col)
     of the given layer. Layers known: conv1, pool1, conv2, pool2, conv3.
     Architecture mirrored:
       conv1  5×5 pad=2 stride=1  => same H×W as input
       pool1  2×2 stride=2
       conv2  5×5 pad=2 stride=1
       pool2  2×2 stride=2
       conv3  3×3 pad=1 stride=1
     Effective stride from input to layer:
       conv1: 1, pool1: 2, conv2: 2, pool2: 4, conv3: 4.
     Effective receptive size in input pixels (per the formula):
       conv1: 5, pool1: 6, conv2: 14, pool2: 16, conv3: 24.
     All centred. Returns {top, left, size}. */
  function receptiveField(layer, row, col) {
    const T = {
      conv1: { stride: 1, size: 5 },
      pool1: { stride: 2, size: 6 },
      conv2: { stride: 2, size: 14 },
      pool2: { stride: 4, size: 16 },
      conv3: { stride: 4, size: 24 },
    };
    const t = T[layer];
    if (!t) throw new Error('Unknown layer: ' + layer);
    // Centre of receptive field in input coords:
    const cy = row * t.stride + (t.stride - 1) / 2;
    const cx = col * t.stride + (t.stride - 1) / 2;
    const top = Math.round(cy - t.size / 2 + 0.5);
    const left = Math.round(cx - t.size / 2 + 0.5);
    return { top, left, size: t.size };
  }

  /* Run the full shapelets28 forward pass given DATA.shapelets weights.
     Returns intermediate feature maps for inspection by scenes. */
  function shapeletsForward(input28x28, weights) {
    if (!weights) return null;
    const x = [input28x28]; // [1][28][28]
    const c1 = relu3D(multiConv2d(x, weights.conv1.kernels, weights.conv1.biases, 2));   // 8×28×28
    const p1 = maxpool3d(c1, 2);                                                          // 8×14×14
    const c2 = relu3D(multiConv2d(p1, weights.conv2.kernels, weights.conv2.biases, 2));  // 16×14×14
    const p2 = maxpool3d(c2, 2);                                                          // 16×7×7
    const c3 = relu3D(multiConv2d(p2, weights.conv3.kernels, weights.conv3.biases, 1));  // 24×7×7
    return { conv1: c1, pool1: p1, conv2: c2, pool2: p2, conv3: c3 };
  }

  window.CNN = {
    zeros2D, zeros3D,
    conv2d, multiConv2d,
    relu2D, relu3D,
    maxpool2d, maxpool3d,
    extractPatch, dotProductBreakdown,
    range2D, receptiveField,
    shapeletsForward,
  };
})();
