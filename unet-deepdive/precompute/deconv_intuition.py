"""B3 — Deconv intuition data for scene 7.

Three artifacts in one file:
  1. Stamp traces. A 3×3 input is multiplied by a 3×3 filter and stamped into
     a 5×5 output. Frames[k] is the canvas after the k-th input cell
     (in row-major order) has stamped. Frame 0 is empty; frame 9 is the
     final summed canvas.
  2. Zero-insert equivalence. The same 3×3 input is "zero-inserted" — placed
     at every other position in a 5×5 grid (corners) — then convolved with
     the same 3×3 filter (padding=1) to give a 5×5 output. This output must
     equal frame 9 of the stamp trace exactly.
  3. 1D matrix-form example. A 4-vector input with a 3-tap filter, transposed
     conv stride=2 -> 7-vector output. We dump the explicit 7×4 W^T matrix
     plus input, filter, output.

For (1) and (2) we use stride=1 transposed conv, which is the mental-model
case. The output is (3 + 3 - 1) = 5 wide.

Output: artifacts/deconv_traces.json
"""

import json
import os
import numpy as np

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)
ROUND = 4


def r(arr):
    return np.round(np.asarray(arr, dtype=np.float64), ROUND).tolist()


def stamp_traces(input_3, filt_3):
    """Manual stamp: each input cell (i,j) stamps `filt * input[i,j]` into the
    5×5 output at rows i..i+2, cols j..j+2. We emit 10 frames: frame 0 is
    empty, frame k (k=1..9) is the canvas after the k-th cell stamped.
    """
    frames = []
    canvas = np.zeros((5, 5), dtype=np.float32)
    frames.append(canvas.copy())
    stamp_meta = []
    k = 0
    for i in range(3):
        for j in range(3):
            v = input_3[i, j]
            contrib = v * filt_3
            canvas[i:i + 3, j:j + 3] += contrib
            frames.append(canvas.copy())
            stamp_meta.append(dict(
                step=k,
                input_cell=[int(i), int(j)],
                input_value=float(v),
                stamp=r(contrib),
                target_rows=[int(i), int(i + 3)],
                target_cols=[int(j), int(j + 3)],
            ))
            k += 1
    return frames, stamp_meta


def zero_insert_equivalent(input_3, filt_3):
    """Pad the 3×3 input with one zero ring -> 5×5, then convolve with the
    3×3 filter (no further padding). Result: 5 - 3 + 1 = 3.

    Wait — that's wrong. The actual zero-insert mental model for *stride-1*
    transposed conv just full-pads the input (pad K-1 on every side, here 2)
    and convolves with the *flipped* kernel under valid mode. The result is
    (3 + 2*2 - 3 + 1) = 5.

    More directly: for stride=1, t-conv is equivalent to convolution
    (cross-correlation) of the input padded with K-1 zeros on each side,
    with the filter flipped. Easiest way: just compute the t-conv result
    directly via the naive sum, which gives the *same* number as the stamp
    trace. We verify equivalence by recomputing it via padding+flip+convolve.
    """
    # Method: scipy-free 2D convolution via numpy.
    H = K = 3
    # Pad input with K-1=2 zeros on every side -> shape 7×7.
    padded = np.zeros((H + 2 * (K - 1), H + 2 * (K - 1)), dtype=np.float32)
    padded[K - 1:K - 1 + H, K - 1:K - 1 + H] = input_3
    # Flip the kernel for true convolution (vs cross-correlation).
    kf = filt_3[::-1, ::-1]
    out_h = padded.shape[0] - K + 1   # 5
    out = np.zeros((out_h, out_h), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_h):
            patch = padded[i:i + K, j:j + K]
            out[i, j] = float(np.sum(patch * kf))
    return out, padded


def matrix_form_1d():
    """1D transposed conv with kernel [a,b,c]=[1,2,3], stride=2.

    Input (4,) -> Output (4*2 + 1 = 9? actually for stride=2 with K=3 and no
    padding, output = (N-1)*S + K = 3*2 + 3 = 9). The PLAN says 7-vector;
    we use stride=2, K=3, padding=1, output_padding=0 -> output_length =
    (N-1)*S - 2*P + K + output_padding = 3*2 - 2 + 3 = 7.

    We dump the explicit 7×4 matrix W^T and input/output vectors so the JS
    can render the matrix.
    """
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    k = np.array([1, 2, 3], dtype=np.float32)
    N = 4
    K = 3
    S = 2
    P = 1
    Lout = (N - 1) * S - 2 * P + K  # 7
    # Build W^T: each input position i contributes its filter into output
    # at columns i*S - P + (0..K-1).
    WT = np.zeros((Lout, N), dtype=np.float32)
    for i in range(N):
        for kk in range(K):
            row = i * S - P + kk
            if 0 <= row < Lout:
                WT[row, i] = k[kk]
    y = WT @ x
    return dict(
        input=r(x),
        filter=r(k),
        output=r(y),
        WT=r(WT),
        stride=S,
        padding=P,
        out_length=int(Lout),
    )


def main():
    input_3 = np.array([
        [0, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    filt = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.float32)  # the "+"

    frames, stamp_meta = stamp_traces(input_3, filt)
    final_frame = frames[-1]
    zi_out, zi_padded = zero_insert_equivalent(input_3, filt)

    # Verify equivalence.
    diff = np.max(np.abs(final_frame - zi_out))
    print(f'Stamp vs zero-insert max abs diff: {diff:.6e}')
    assert diff < 1e-5, 'Stamp and zero-insert should agree exactly.'

    matrix_1d = matrix_form_1d()

    bundle = dict(
        input=r(input_3),
        filter=r(filt),
        frames=[r(f) for f in frames],
        stamp_meta=stamp_meta,
        zero_insert=dict(
            padded_input=r(zi_padded),
            output=r(zi_out),
        ),
        matrix_1d=matrix_1d,
    )
    out_path = os.path.join(ART, 'deconv_traces.json')
    with open(out_path, 'w') as f:
        json.dump(bundle, f, separators=(',', ':'))
    print(f'Wrote {out_path}  (frames={len(frames)}, equivalence verified)')


if __name__ == '__main__':
    main()
