# U-Net deepdive

A 15-scene browser visualization that builds intuition for the U-Net
used in semantic segmentation: per-pixel prediction, transposed
convolution, skip connections, the assembled network, training
dynamics, and failure modes. Sibling of `cnn-deepdive/`. See `PLAN.md`
for the scene-by-scene design.

## Viewing

Plain static HTML/CSS/JS, no build step. From this directory:
```
python3 -m http.server 8000
# then open http://localhost:8000/ in a browser
```
Dot-pager, prev/next buttons, or arrow keys navigate scenes; within a
scene, arrows first advance the internal step engine. Top-right button
(or `t`) toggles light/dark theme. URL flags: `#scene=N` jumps to a
scene, `#scene=N&run` auto-advances its steps.

## Regenerating the precompute

All artifacts behind `data/datasets.js` (~4.1 MB) come from
`precompute/`. Rebuild from scratch with (needs `numpy` + `torch`,
runs in a few minutes on CPU):
```
bash precompute/run_all.sh
```

The no-skip pixel accuracy quoted in scene 9 (95.40% vs. 99.98% with
skips) is from a deliberately undertrained no-skip baseline (3 epochs
vs. 30); identical training reaches ~99.6% on this easy cartoon
dataset and erases the visual contrast scene 9 depends on.
