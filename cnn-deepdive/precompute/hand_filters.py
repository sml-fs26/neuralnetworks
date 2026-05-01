"""The 8 hand-designed 5x5 filters used in scene 1 of the visualization.

Spec (from the brief):
  vertical:   center column +2, sides -1, others 0          (column pattern)
  horizontal: middle row +2, top/bottom -1                  (row pattern)
  diag_down:  main diagonal +2, others -1                   (\ pattern)
  diag_up:    anti-diagonal +2, others -1                   (/ pattern)
  dot:        center +4, inner ring +1, outer ring -1       (centered blob)
  ring:       inner ring +2, center -2, outer 0/-1          (donut)
  top_half:   top 2 rows +1, middle 0, bottom 2 rows -1
  left_half:  left 2 cols +1, middle 0, right 2 cols -1
"""

import numpy as np


def build():
    F = {}

    # vertical: center column (col index 2) +2, sides (cols 0,1,3,4) -1, others 0
    v = np.zeros((5, 5), dtype=np.float32)
    v[:, 2] = 2.0
    v[:, [0, 1, 3, 4]] = -1.0
    F['vertical'] = v

    # horizontal: middle row +2, top/bottom rows -1, middle inner rows 0
    h = np.zeros((5, 5), dtype=np.float32)
    h[2, :] = 2.0
    h[[0, 4], :] = -1.0
    F['horizontal'] = h

    # diag_down: main diagonal (top-left to bot-right) +2, others -1
    dd = -np.ones((5, 5), dtype=np.float32)
    for i in range(5):
        dd[i, i] = 2.0
    F['diag_down'] = dd

    # diag_up: anti-diagonal +2, others -1
    du = -np.ones((5, 5), dtype=np.float32)
    for i in range(5):
        du[i, 4 - i] = 2.0
    F['diag_up'] = du

    # dot: center +4, inner ring +1, outer ring -1
    dot = np.zeros((5, 5), dtype=np.float32)
    dot[2, 2] = 4.0
    # inner ring (Chebyshev distance 1 from center)
    for i in range(5):
        for j in range(5):
            d = max(abs(i - 2), abs(j - 2))
            if d == 1:
                dot[i, j] = 1.0
            elif d == 2:
                dot[i, j] = -1.0
    F['dot'] = dot

    # ring: inner ring +2, center -2, outer ring 0
    ring = np.zeros((5, 5), dtype=np.float32)
    ring[2, 2] = -2.0
    for i in range(5):
        for j in range(5):
            d = max(abs(i - 2), abs(j - 2))
            if d == 1:
                ring[i, j] = 2.0
            elif d == 2:
                ring[i, j] = 0.0
    F['ring'] = ring

    # top_half: top 2 rows +1, middle row 0, bottom 2 rows -1
    th = np.zeros((5, 5), dtype=np.float32)
    th[0:2, :] = 1.0
    th[3:5, :] = -1.0
    F['top_half'] = th

    # left_half: left 2 cols +1, middle col 0, right 2 cols -1
    lh = np.zeros((5, 5), dtype=np.float32)
    lh[:, 0:2] = 1.0
    lh[:, 3:5] = -1.0
    F['left_half'] = lh

    return F


if __name__ == '__main__':
    F = build()
    for k, v in F.items():
        print(k)
        print(v)
