# U-Net deepdive — build plan

This folder holds the plan for a new visualization, sibling to `cnn-deepdive/`,
dedicated to building deep intuition for the U-Net architecture used in
semantic segmentation. The CNN deepdive ends with a *single* segmentation
scene (`cnn-deepdive/js/scenes/scene9.js`) that shows the U-shape and a few
predictions. That scene is a teaser. This deepdive is the full treatment:
dataset, encoder, bottleneck, upsampling, transposed convolution, skip
connections, the assembled architecture, training, and failure modes — each
with its own scene and its own intuition-building visuals.

The intent matches the level of care taken in `cnn-deepdive/`: every concept
that is non-obvious gets a scene that makes it obvious, with a small,
clickable, animated demo and a precomputed numerical artifact backing it.

---

## 0. Scope and ground truth

### 0.1 What this visualization is teaching

A learner who has watched `cnn-deepdive/` knows what a filter is, what a
feature map is, what pooling does, and what a stack of conv-units produces.
What they do *not* yet know:

1. **Why segmentation is harder than classification** — output shape, loss
   shape, what "per-pixel prediction" means.
2. **What the segmentation dataset looks like** — pairs of `(image, label_map)`
   instead of `(image, single_label)`, and how that label map is encoded
   (an `H×W` int tensor of class indices).
3. **Upsampling** — three flavors (nearest-neighbor, bilinear, transposed
   conv) and the question each one answers.
4. **Transposed convolution** ("deconvolution") — the **stamp-and-sum**
   intuition, contrasted with the **slide-and-dot-product** of regular conv.
   Plus the zero-insertion equivalence and the gradient-of-conv interpretation.
5. **Skip connections** — *why* the U-Net has them. The "what got lost"
   demonstration: after three pools, boundary pixels are mush; the encoder
   skip restores spatial detail that the bottleneck threw away.
6. **Concatenation skips vs. residual additions** — disambiguation. The
   existing U-Net uses **concatenation** (`torch.cat([u, e], 1)`,
   `cnn-deepdive/precompute/train_segmenter.py:51,53`); ResNet uses **addition**
   (`x + F(x)`). They are different operators with different motivations.
   Both belong to the family of "shortcut connections". The user's question
   ("why is there a residual layer in the U-Net") is a common conflation we
   must address head-on.
7. **The full U-Net assembled** — encoder + bottleneck + decoder + skips +
   1×1 head, with shapes and channel counts visible at every step.
8. **Training** — pixel-wise cross-entropy, what the loss surface "sees", and
   how the prediction map evolves over epochs.
9. **Failure modes** — where the model is right, where it is wrong, and what
   classes of error the architecture struggles with.

### 0.2 The U-Net we are explaining

Reused from `cnn-deepdive/precompute/train_segmenter.py:32-57`. We do not
retrain from scratch unless we need new artifacts; we re-export intermediate
activations and build new visualizations on top.

```
TinyUNet(n_classes=5):
  enc1 = Conv2d(3,16,3,p=1) -> ReLU -> Conv2d(16,16,3,p=1) -> ReLU      # 64×64×16
  pool1 = MaxPool2d(2)                                                    # 32×32×16
  enc2 = Conv2d(16,32,3,p=1) -> ReLU -> Conv2d(32,32,3,p=1) -> ReLU     # 32×32×32
  pool2 = MaxPool2d(2)                                                    # 16×16×32
  enc3 = Conv2d(32,64,3,p=1) -> ReLU -> Conv2d(64,64,3,p=1) -> ReLU     # 16×16×64  <- bottleneck
  up2  = ConvTranspose2d(64,32,2,stride=2)                                # 32×32×32
  dec2 = conv_block(64,32)   on  cat([up2, enc2], dim=1)                  # 32×32×32
  up1  = ConvTranspose2d(32,16,2,stride=2)                                # 64×64×16
  dec1 = conv_block(32,16)   on  cat([up1, enc1], dim=1)                  # 64×64×16
  out  = Conv2d(16,5,1)                                                   # 64×64×5  (logits)
```

Loss: pixel-wise cross-entropy summed over all `64*64 = 4096` pixels per image.
Classes (0..4): `sky, grass, sun, tree, person`.

### 0.3 Dataset

Reused from `cnn-deepdive/precompute/scene64_data.py`: 64×64 cartoon scenes
with a gradient sky, a green grass strip, and randomly placed sun, tree,
and person silhouettes. Label maps are dense `H×W` int arrays. We have the
generator, the trained model, and a 100-sample test split already
materialized in `cnn-deepdive/precompute/artifacts/scene64_data.npz` and
`segmenter.pt`. The new viz can copy or symlink these, or rerun the precompute
with deeper instrumentation.

### 0.4 Non-goals

- We are **not** building a real-world segmenter (no Cityscapes, no medical
  imaging). The cartoon dataset is *deliberately* tiny and synthetic so every
  concept can be shown end-to-end in the browser with no GPU.
- We are **not** introducing batch norm, dropout, attention, or any U-Net
  variant beyond the original concat-skip design. ResNet-style residuals get
  exactly one scene of comparison, no more.
- We are **not** rebuilding the chrome (header, pager, theme toggle). We copy
  the proven shell from `cnn-deepdive/`.

---

## 1. File layout

Mirror `cnn-deepdive/` exactly so the chrome, drawing helpers, and step
engine all transfer with minimal edits.

```
unet-deepdive/
├── PLAN.md                       (this file)
├── README.md                     (one-paragraph orientation, written last)
├── index.html                    (loads css + js + scenes, in order)
├── css/
│   ├── style.css                 (copied from cnn-deepdive, then trimmed)
│   ├── scene0.css ... sceneN.css (per-scene styling)
├── js/
│   ├── theme.js                  (copy from cnn-deepdive)
│   ├── drawing.js                (copy + add: paintLabelMap, paintRGB,
│   │                              paintFeatureCard helpers extracted from
│   │                              cnn-deepdive scene9)
│   ├── unet.js                   (NEW: small JS forward-pass utilities and
│   │                              array helpers — analogous to cnn.js)
│   ├── katex-helpers.js          (copy from cnn-deepdive)
│   ├── main.js                   (copy from cnn-deepdive, replace SCENES list)
│   └── scenes/
│       ├── scene0.js
│       ├── scene1.js
│       ├── ...
│       └── sceneN.js
├── data/
│   └── datasets.js               (window.DATA = { scene64, weights, ... })
├── precompute/
│   ├── scene64_data.py           (symlink or copy of cnn-deepdive's)
│   ├── train_segmenter.py        (symlink or copy)
│   ├── export_to_js.py           (NEW: dumps all artifacts the new scenes
│   │                              need — see §3)
│   ├── upsample_demos.py         (NEW: precomputes nearest/bilinear/transposed
│   │                              conv outputs on a few small inputs)
│   ├── deconv_intuition.py       (NEW: precomputes stamp/spray figures and
│   │                              the "zero-insert + conv" equivalence)
│   ├── skip_ablation.py          (NEW: trains a U-Net WITHOUT skips, exports
│   │                              its predictions for the side-by-side scene)
│   ├── training_traces.py        (NEW: snapshots intermediate predictions
│   │                              every K steps during training)
│   └── artifacts/
└── vendor/
    └── katex/                    (copy from cnn-deepdive)
```

Conventions inherited from `cnn-deepdive`:

- One CSS file per scene, scoped under an `s{N}-` class prefix.
- One JS file per scene, registering `window.scenes.scene{N} = function(root){...}`.
- Step engine: each scene has a `state.step`, `applyStep`, `render`, and
  click/keyboard navigation. `render()` is idempotent.
- `&run` URL flag auto-advances the scene. `?scene=N` jumps to a scene.
- Light/dark theme via CSS variables. All colors come from CSS, never
  hard-coded in JS (so theme switching just works).
- Drawing utilities go in `drawing.js`; numerical utilities in `unet.js`.

---

## 2. Scene list

15 scenes total. Each gets a one-line title, a short description, the
specific intuition it builds, and the visuals required.

> Numbering convention: zero-indexed, contiguous, matches the array index.
> Files named `scene{N}.js` so reordering does not rename files (per the
> comment at `cnn-deepdive/js/main.js:10-13`).

### Scene 0 — A U-Net, end to end

**Title:** "A U-Net, end to end"
**Goal:** orient the viewer. Show the full U on the left, an input image
flowing through it on the right, and the final segmentation appearing.
A single auto-advance pulse (similar to scene0 of CNN deepdive).

**Visuals:**
- Hero text + lede ("a classifier says 'this image is a circle'; a segmenter
  says 'this pixel is a tree, this strip is grass'").
- The full U diagram (encoder column, bottleneck, decoder column, skip arcs,
  1×1 conv head). Static, all cards filled.
- Input → output strip: the 64×64 RGB input on the left, the 64×64 colored
  label map on the right.
- A pulsing "play" button that does a 3-second forward sweep highlighting
  encoder → bottleneck → decoder → skips → output, in sequence.

**Step engine:** 0 = static U; 1 = sweep encoder; 2 = sweep bottleneck;
3 = sweep skip arcs; 4 = sweep decoder; 5 = reveal output. `&run` plays
0 → 5 in 3 seconds.

**Caption arc:** "We will spend the next 14 scenes building this. By the
end you should know why every line of this diagram is here."

---

### Scene 1 — Classification vs. segmentation

**Title:** "From one label to a label per pixel"
**Goal:** make the *output shape difference* viscerally obvious. This is the
pivot from the CNN deepdive's classification mindset.

**Visuals:**
- Two side-by-side panels.
- Left: a 64×64 cartoon input → CNN classifier → vector of 5 probabilities
  → single label "scene contains a tree". One number.
- Right: the same input → U-Net segmenter → a 64×64×5 tensor of probabilities
  → a 64×64 label map. One number per pixel. 4096 numbers.
- Hover over the right output: each pixel highlights to show "this pixel got
  its own 5-way softmax, and the argmax is `tree`".
- Inset: "loss for the classifier = cross-entropy on 1 distribution.
   loss for the segmenter = cross-entropy summed over 4096 distributions."

**Step engine:** 0 = both outputs blank; 1 = reveal classifier output;
2 = reveal segmenter output (label map fades in); 3 = hover-pixel mode
enabled; 4 = the two cross-entropy formulas appear side-by-side
(rendered with KaTeX).

**Caption arc:** "Same image. Same convolutional machinery. Different
*shape* of answer. Everything else in this deepdive follows from that
shape change."

---

### Scene 2 — The dataset

**Title:** "What a segmentation dataset looks like"
**Goal:** show what the model is being shown. Viewer must understand
that supervision is *dense*: every pixel has a known correct class.

**Visuals:**
- A grid of 12 sample images and their label maps, side by side. Each pair
  shares a frame.
- A "data-generator" inset: click a button to draw a new sample live
  (calls a JS reimplementation of the simplest pieces of `scene64_data.py`,
  or just rotates through 50 precomputed samples).
- Class legend (5 swatches: sky, grass, sun, tree, person).
- Histogram strip: how often each class appears across all 600 training
  samples. Sky and grass dominate; sun is rare. (Foreshadows class
  imbalance, which the failure-modes scene revisits.)

**Step engine:** 0 = grid only; 1 = legend appears; 2 = data-generator
button is enabled; 3 = histogram fades in.

**Caption arc:** "600 training images, 100 test. Each image is 64×64×3 RGB,
each label is 64×64 ints in {0..4}. The labels were placed by the
generator; in real life a human would draw them, pixel by pixel, on a
graphics tablet."

---

### Scene 3 — Why a plain CNN classifier won't do

**Title:** "Where does the resolution go?"
**Goal:** motivate the existence of a decoder. After three pools, the
spatial map is 8×8. We want a 64×64 prediction. Something has to put the
resolution back.

**Visuals:**
- Repurpose the encoder visualization from `cnn-deepdive` scene 4-5: input
  → enc1 (64×64) → pool → enc2 (32×32) → pool → enc3 (16×16). Show the
  feature maps shrinking.
- Then show: "and now we want a 64×64 answer". Big arrow pointing back up
  to the input resolution. Arrow has a giant "?" on it.
- A failed alternative: "what if we just stretched the 16×16 prediction
  back up to 64×64 with nearest-neighbor?" Show that prediction. It's
  blocky, with 4×4 squares of single colors. Hopeless for thin objects
  like the tree trunk.

**Step engine:** 0 = encoder shrinkage; 1 = the "?" arrow; 2 = the failed
nearest-neighbor blow-up.

**Caption arc:** "Pooling threw away spatial detail. We can pretend it
didn't, and stretch the 16×16 answer back up — but the result has no
spatial precision. The decoder is what gives it back."

---

### Scene 4 — The encoder, in one breath

**Title:** "The encoder is the CNN you already know"
**Goal:** brisk reuse, not a rebuild. Encoder = familiar CNN territory.

**Visuals:**
- Three encoder-level cards (64×64×16, 32×32×32, 16×16×64).
- For one selected sample: light up each level in sequence, showing 4
  representative channels per level (similar to scene9 in the CNN deepdive).
- Annotated with "doubling channels, halving resolution — the trade we
  established in the CNN deepdive."

**Step engine:** 0 = input only; 1 = enc1 lit; 2 = enc2 lit; 3 = enc3 lit
(the bottleneck).

**Caption arc:** "If you watched the CNN deepdive, this is the encoder you
already understand. The interesting half of a U-Net is the *other* half."

---

### Scene 5 — The bottleneck

**Title:** "16×16 of pure semantics"
**Goal:** make the viewer see what the bottleneck *is*: a small map where
each cell looks at a large patch of the input and encodes "what kind of
stuff is here".

**Visuals:**
- The 16×16×64 bottleneck. Show 16 of its 64 channels as a 4×4 thumbnail
  grid. For one selected channel, click any cell and see the receptive
  field highlighted on the input — a roughly 30×30 patch.
- A sliding panel: pick three semantically distinct regions (sky, grass,
  tree) and see that the bottleneck's activation profile differs across
  them. "These 64 numbers per cell are a 64-dim semantic fingerprint."

**Step engine:** 0 = bottleneck grid; 1 = receptive-field overlay enabled;
2 = three-region comparison panel.

**Caption arc:** "By the time we get here, each cell has a giant receptive
field but no spatial precision. The cell knows roughly *what* is there.
It does not know exactly *where*. Fixing that 'where' is the decoder's job."

---

### Scene 6 — Upsampling: three ways to grow a map

**Title:** "How do you make a map bigger?"
**Goal:** establish the design space *before* introducing the learned answer.
The viewer should leave this scene with the question "but those are just
heuristics — couldn't we *learn* the upsample?" Scene 7 then answers it.

**Visuals:**
- A small input feature map (e.g., 4×4 single channel) with a few non-zero
  cells.
- Three side-by-side panels, each grows it to 8×8 by a different rule:
  - **Nearest-neighbor**: each input cell becomes a 2×2 block of itself.
  - **Bilinear**: weighted average of the four nearest known cells.
  - **Strided transposed conv (preview)**: shown as a black box for now —
    "we'll open this in the next scene".
- For each, the cells light up showing which input cells contributed.
- Cost / property table beneath:
  | method            | learnable? | smooth? | sharp boundaries? |
  | nearest-neighbor  | no         | no      | yes (blocky)      |
  | bilinear          | no         | yes     | no (blurry)       |
  | transposed conv   | **yes**    | depends | depends           |

**Step engine:** 0 = input map; 1 = nearest-neighbor reveal; 2 = bilinear;
3 = transposed-conv black box; 4 = the cost/property table.

**Caption arc:** "Nearest-neighbor is fast and dumb. Bilinear is smooth and
dumb. Both have *zero learnable parameters*. The next scene introduces an
upsampling operator with *learnable* parameters: the transposed convolution.
That is the operator the U-Net actually uses."

---

### Scene 7 — Transposed convolution (the centerpiece)

**Title:** "Transposed convolution: stamp instead of slide"
**Goal:** the conceptual highlight of the deepdive. Build *three*
complementary intuitions:

1. **Stamp/spray intuition.** A regular conv slides a filter across the
   input and computes a dot-product per position. A transposed conv does
   the *opposite*: each *input* pixel is multiplied by the entire filter
   and *stamped* into the output. Where stamps overlap, they sum.
2. **Zero-insertion equivalence.** A stride-2 transposed conv with kernel
   `K` is equivalent to: insert zeros between the input pixels, then run
   a regular conv with kernel `K`. Same result, different mental model.
3. **Gradient-of-conv interpretation** (optional advanced inset, skippable).
   The transposed conv is the operator that propagates gradients through
   a regular conv. The name "transposed" comes from the fact that, written
   as matrix multiplications, conv is `y = W x` and transposed conv is
   `x' = W^T y`. The advanced inset shows this for a tiny example with
   actual matrices.

**Visuals:**
- Top: a 3×3 input (a few highlighted cells) and a 3×3 filter (a small
  picture).
- Middle: the **stamp animation**. Each input cell, in sequence, multiplies
  the filter and "stamps" the result into a corresponding region of a
  6×6 output. Stamps overlap; cells sum. Animate a running tally.
- Toggle: switch from stamp animation to **zero-insert animation**. Show
  the input being interleaved with zeros (3×3 → 6×6 sparse map), then a
  3×3 filter sliding over it as a regular conv. Final output identical to
  the stamp version. Insert a "see, same answer" overlay.
- Bottom: a 1D toy version (a 4-vector convolved with a 2-tap filter,
  transposed) with the explicit matrix `W^T` shown via KaTeX. This is the
  optional advanced inset; collapsed by default behind a "show me the
  matrix" affordance.
- Color the output cells by which input cell contributed (or by which
  multiple input cells contributed).
- Two interactive knobs:
  - **Stride** (1 or 2 — show what happens to output size).
  - **Filter pattern** (preset patterns: a "+", an edge, a checkerboard).
    Changing the filter changes the stamp shape live.

**Step engine:** 0 = setup, input + filter shown; 1 = stamp the first cell;
2 = stamp the second cell, sum visible; 3 = continue stamping until
output is complete; 4 = toggle to zero-insert view; 5 = show the matrix
formulation (advanced); 6 = enable the stride and filter knobs.

**Caption arc:** "A regular convolution looks at the input and asks, 'how
much of this pattern is here?'. A transposed convolution does the dual:
each input value asks, 'where in the output should I leave my mark?', and
stamps the filter pattern there. Wherever stamps overlap, they sum. This
is *learned upsampling*: the filter is a parameter trained by gradient
descent, just like every other filter in the network."

> **Implementation note for the agent.** This scene is the largest and
> most pedagogically dense in the deepdive. It will need its own
> precompute (numerical stamp traces for several filters), its own CSS
> (the matrix view in the advanced inset), and probably 250+ lines of JS.
> Budget accordingly. See §4 for the suggestion to delegate this to a
> dedicated agent that itself dispatches sub-agents for (a) the stamp
> animation, (b) the zero-insert view, (c) the matrix inset.

---

### Scene 8 — The decoder, walked through

**Title:** "Walking up the right side of the U"
**Goal:** mirror scene 4 (the encoder) but on the way back up. Show the
upsample step and the conv-block step at each decoder level.

**Visuals:**
- Two decoder cards (32×32×32 → 64×64×16) on the right column.
- For one selected sample, animate: bottleneck (16×16×64) → up2 (32×32×32,
  via transposed conv) → conv_block → dec2 (32×32×32) → up1 (64×64×16) →
  conv_block → dec1 (64×64×16). Show the up arrows and the conv arrows
  separately so the two operations are not confused.
- Important: at this point the skip connections are *not yet drawn*. Scene
  9 introduces them.

**Step engine:** 0 = bottleneck only; 1 = up2 lit; 2 = dec2 lit (after
conv-block); 3 = up1 lit; 4 = dec1 lit; 5 = output (1×1 conv → softmax →
label map). Note that since skips are not yet present, the prediction
will be visibly worse than the final U-Net's. That bad prediction is the
hook for scene 9.

**Caption arc:** "Up, conv, up, conv. The decoder is the encoder run
backwards. But notice the prediction. It is 'kind of right' but the
boundaries are mush. Hold on to that picture — the next scene explains
why, and what to do about it."

---

### Scene 9 — Skip connections: what got lost

**Title:** "What the bottleneck threw away"
**Goal:** make the *necessity* of skip connections visceral. Side-by-side:
no-skip prediction vs. with-skip prediction. The viewer should see, with
their own eyes, that the skip connection rescues the boundaries.

**Empirical note (resolved during B4 training).** On the cartoon dataset,
both with-skip and no-skip models hit ~99.6–99.98% pixel accuracy in
aggregate. The dataset is too easy (16×16×64 = 16384-number bottleneck >
4096-pixel input) for the no-skip model to fail dramatically. **However**,
the no-skip model's ~0.4% errors are concentrated on 1-pixel-wide object
boundaries — exactly the place skip connections are supposed to help.
The framing of this scene must therefore be **boundary precision**, not
aggregate accuracy. Specifically:
- Lead with diff overlays, not accuracy numbers (the numbers look the
  same to two decimal places).
- Include a 4× zoom-in viewer focused on the tree trunk and person
  silhouette of one selected sample. The no-skip model's boundary fuzz
  will be plainly visible at zoom; the with-skip model's will be
  pixel-perfect.
- One honesty caption: "On harder data with finer features, the gap
  becomes much larger. Here, both models hit ~99.6%+, but the
  *no-skip errors are all on boundaries* — exactly the spatial detail
  skips are designed to preserve."

**Visuals:**
- Top row: input image, ground-truth label, prediction *without* skips,
  prediction *with* skips. Four 256×256 panels. The "without" panel uses
  a separately trained no-skip baseline (see `precompute/skip_ablation.py`).
- Diff overlay: outline pixels where each prediction disagrees with the
  ground truth. The no-skip diff is dramatically denser, especially around
  thin objects (tree trunk, person's body).
- Below: a "decoder input shapes" diagram. Shows that without skips, dec2
  has 32 channels (just up2). With skips, dec2 has 32 + 32 = 64 channels
  (up2 cat enc2). Same for dec1. The skip *concatenates* — it does not
  overwrite or add.
- Hover any pixel of the no-skip prediction to highlight which receptive
  field in the bottleneck "owns" it. Then toggle to with-skip and see
  the same pixel now also "owns" a much smaller receptive field from
  enc1 — the source of the sharp boundary information.

**Step engine:** 0 = input + GT only; 1 = no-skip prediction reveals;
2 = with-skip prediction reveals (the contrast is the punchline);
3 = diff overlays appear; 4 = channel-count diagram; 5 = receptive-field
hover mode enabled.

**Caption arc:** "The bottleneck has rich semantics but coarse spatial
precision. The encoder *had* sharp spatial precision earlier, before any
pooling. The skip connection grabs those early, sharp feature maps and
hands them to the decoder, so the decoder can have its cake (deep
semantics from the bottleneck path) and eat it (sharp boundaries from
the skip path)."

---

### Scene 10 — Concatenation skip vs. residual addition

**Title:** "Skip vs. residual: two cousins, different jobs"
**Goal:** disambiguate. The user's question — "why is there a residual
layer in the U-Net" — is a common conflation we resolve here.

**Visuals:**
- Two diagrams side by side.
- Left: **U-Net concat skip.** Two tensors of shape `(C, H, W)` and
  `(C', H, W)` are *concatenated along the channel dim* into `(C+C', H, W)`,
  then convolved. The decoder *learns* how to mix them; nothing forces it
  to keep both, but it has access to both.
- Right: **ResNet residual add.** `y = F(x) + x`. The output has the same
  shape as `x`. The shortcut path forces the block to learn a *delta*
  on top of the input. The motivations are different: residuals are
  primarily about **gradient flow** in very deep networks and giving
  every block an "identity" option to fall back on. Skips in U-Net are
  primarily about **handing spatial detail across the bottleneck**.
- A small live demo: a tiny scalar example. Two 1-channel `8×8` "feature"
  inputs `a` and `b`.
  - **Concat-then-conv** result is shown when the decoder learns
    coefficients `(α, β)`: output ≈ `α*a + β*b`. Move sliders for `α, β`.
  - **Add** result is just `a + b` — no slider. The viewer feels the
    difference.
- A short comparison table at the bottom:
  | property                  | U-Net concat skip | ResNet residual add |
  | output channels           | C + C'            | C                    |
  | requires same shape?      | same H, W         | same H, W *and* C    |
  | learnable mixing?         | yes (via decoder) | no (fixed +)         |
  | primary motivation        | spatial detail    | gradient flow        |

**Step engine:** 0 = both diagrams shown; 1 = the "shapes" annotations
appear; 2 = the live demo enables; 3 = the comparison table reveals.

**Caption arc:** "These are two different operators in the same family of
'shortcut connections'. People sometimes call any shortcut a 'residual
connection', but in the U-Net the operator is concatenation, and the
purpose is to put spatial detail back into the decoder. ResNet's residual
addition is a different beast for a different problem."

---

### Scene 11 — The full U-Net, assembled

**Title:** "The U-Net, fully wired"
**Goal:** the synthesis scene. Every previous concept made small and clear;
now they are placed in a single diagram with shapes and channels labeled
on every edge. The viewer can read the entire network in one image.

**Visuals:**
- A large U-shape diagram with every edge labeled `H × W × C`.
- Every operation labeled: conv-block, max-pool, transposed conv, concat,
  conv-block, 1×1 conv, softmax.
- Hover any node to see the formula and shape transformation
  (rendered with KaTeX).
- Hover any *edge* (skip arc) to see the source and destination tensor
  shapes and the concatenation arithmetic ("16 + 16 → 32").
- A single "play forward" button animates a real input flowing through
  all of it, lighting up each operation in temporal order, ending at the
  predicted label map.

**Step engine:** 0 = static diagram, no labels; 1 = shape labels appear on
edges; 2 = operation labels on nodes; 3 = hover mode enabled;
4 = "play forward" sweep.

**Caption arc:** "Every line in this diagram has now been earned. The
encoder you knew. The bottleneck you saw. The upsample is a transposed
conv — scene 7. The skip is a concatenation — scenes 9 and 10. The 1×1
conv at the head turns the final 16-channel feature map into 5 class
logits, one per pixel."

---

### Scene 12 — Training: per-pixel cross-entropy

**Title:** "The loss that knows about every pixel"
**Goal:** show what training the U-Net actually optimizes, and why it is
just the classification loss summed over pixels.

**Visuals:**
- Pick one training sample. Show the prediction (a 64×64×5 tensor of
  probabilities) on the left.
- For one selected pixel, expand the 5-way softmax bar chart and the
  ground-truth one-hot. Compute the cross-entropy for that pixel
  (KaTeX-rendered formula).
- Then aggregate: a small heatmap of per-pixel cross-entropy across the
  whole 64×64. Bright spots are pixels the model is currently uncertain
  or wrong about.
- Mean over all pixels = the scalar loss the optimizer sees.
- Hover any pixel of the heatmap to expand its 5-way softmax.

**Step engine:** 0 = prediction + ground-truth; 1 = pick a pixel,
KaTeX formula appears; 2 = aggregate heatmap; 3 = hover mode enabled;
4 = scalar loss number revealed.

**Caption arc:** "It is the same cross-entropy we use for classification,
applied 4096 times and averaged. No new loss machinery. The optimizer's
job has not changed, only the shape of the thing it is optimizing."

---

### Scene 13 — Training dynamics

**Title:** "Watching the segmentation come into focus"
**Goal:** show how predictions evolve as training progresses. This is
the U-Net analogue of `gradient-descent-viz/`'s loss-curve scene, but
with a visual prediction map at each snapshot.

**Visuals:**
- A timeline slider: epoch 0 → epoch 30. Below the slider, a row of
  prediction snapshots for one fixed sample. The earliest predictions
  are mostly "everything is sky"; later predictions get the grass strip
  right; later still, objects emerge with sharp boundaries.
- A loss curve on the right, with a marker tracking the current epoch.
- A second curve below the loss: mean pixel accuracy. Climbs from
  ~0.4 (everything-is-sky baseline) to >0.9 by the end.
- Optional: a "weight magnitude" sparkline for the transposed-conv
  filters specifically, so the viewer can see "the upsamplers are also
  being trained, not just the encoder".

**Step engine:** 0 = epoch-0 frame; 1 = the slider becomes scrubbable;
2 = loss curve overlays; 3 = accuracy curve overlays.

**Caption arc:** "Same gradient descent we have seen before. The
optimizer has 30 epochs to figure out, jointly, how to encode, how to
upsample, how to decode, and what to put at the end. The boundaries
sharpen as the upsamplers learn."

---

### Scene 14 — Where it works, where it fails

**Title:** "What this U-Net gets right, and what it doesn't"
**Goal:** honesty. Every model fails somewhere. The viewer should leave
knowing not just how this works, but where it stumbles, and why.

**Visuals:**
- A gallery of 20 test samples, sorted by prediction accuracy.
- The top row: easy successes (full sky + grass + a single big tree).
- The bottom row: hardest failures. Pick samples where the model
  confuses tree-canopy with grass, or person-body with tree-trunk
  (similar reds/greens), or misses a small sun.
- For each failure, a "what went wrong" annotation: e.g., "the sun is
  smaller than the bottleneck cell — the model literally cannot see it
  at the deepest level".
- Aggregate confusion matrix at the bottom (5×5, normalized rows). The
  viewer sees that sun is the worst class (rare + small), grass is
  perfect (huge + uniform).

**Step engine:** 0 = gallery sorted; 1 = annotations on the failure row;
2 = confusion matrix appears.

**Caption arc:** "The U-Net is not magic. It has a fixed receptive field,
a fixed resolution path, and a fixed training distribution. Tiny objects,
rare classes, and ambiguous color combinations are exactly where it
breaks. Better architectures (more skips, attention, multi-scale
training) are answers to specific failures like these. That is a
different deepdive."

---

## 3. Precompute artifacts

What the JS needs and where it comes from. The agent for §4-B owns this.

### 3.1 Reused as-is from `cnn-deepdive/precompute/artifacts/`

| artifact                  | source script                | used in scene(s)        |
|---------------------------|------------------------------|-------------------------|
| `scene64_data.npz`        | `train_segmenter.py`         | 0, 1, 2, 4, 5, 8, 9, 11, 12, 13, 14 |
| `segmenter.pt`            | `train_segmenter.py`         | (re-run inference for new exports) |

These are copied into `unet-deepdive/precompute/artifacts/` (do not symlink
unless we are sure the cnn-deepdive copy will not be regenerated under
us).

### 3.2 New artifacts

| artifact                              | producer                       | used in scene(s) |
|---------------------------------------|--------------------------------|------------------|
| `unet_intermediates.npz`              | `export_to_js.py`              | 4, 5, 8, 11, 12  |
| `bottleneck_rfields.json`             | `export_to_js.py`              | 5                |
| `upsample_demos.json`                 | `upsample_demos.py`            | 6                |
| `deconv_traces.json`                  | `deconv_intuition.py`          | 7                |
| `noskip_segmenter.pt` + predictions   | `skip_ablation.py`             | 9                |
| `training_traces.npz`                 | `training_traces.py`           | 13               |
| `confusion_matrix.json` + failure ids | `export_to_js.py`              | 14               |

### 3.3 Loading convention

`data/datasets.js` is a single JS file that defines `window.DATA`. Each
artifact above becomes one key on `window.DATA`. Numerical arrays are
emitted as nested JS literals (small) or as base64-encoded Float32Arrays
(large), following whatever cnn-deepdive does in
`cnn-deepdive/data/datasets.js`. Agent §4-B should inspect that file and
match its conventions.

---

## 4. Agent breakdown

The deepdive splits cleanly into 9 work packages. Several are large enough
that the assigned agent should itself **ultrathink** and **fan out to
sub-agents** (called out below). Numbers are rough — what matters is the
parallelization shape, not exact estimates.

### Dependency graph

```
A. Scaffolding ──┐
                 ├──► C. Dataset/task scenes (1, 2)
B. Precompute ───┤
                 ├──► D. Encoder/bottleneck/decoder (4, 5, 8)
                 │
                 ├──► E. Upsampling deep-dive (6)
                 │
                 ├──► F. Transposed convolution (7)              ◄── biggest single piece
                 │
                 ├──► G. Skip connections (9, 10)                ◄── second biggest
                 │
                 ├──► H. Architecture + training (11, 12, 13)
                 │
                 ├──► I. Overview + failure modes (0, 14)
                 │
                 └──► J. Polish + integration
```

`A` and `B` must finish before any scene-building agent starts.
After `A`+`B` are done, `C, D, E, F, G, H, I` can run **in parallel**.
`J` runs last.

### Agent A — Scaffolding (depends on: nothing)

**Scope:** create the file layout, copy the chrome from `cnn-deepdive/`,
adapt `index.html` and `js/main.js` for the new SCENES list, set up
empty per-scene CSS+JS files. Result: navigating to `unet-deepdive/index.html`
shows 15 placeholder scenes that the dot-pager and arrow keys can move
between.

**Specifically copy (do not edit):** `vendor/katex/`, `js/theme.js`,
`js/katex-helpers.js`. **Copy + adapt:** `js/drawing.js` (extract the
`paintRGB`, `paintLabelMap`, `paintFeatureCard`, `paintBlankCard` helpers
out of `cnn-deepdive/js/scenes/scene9.js:89-223` and promote them into
`drawing.js` so every U-Net scene can use them). **Copy + replace
SCENES list:** `js/main.js`, `index.html`. **Trim:** `css/style.css`
(keep the `.hero`, `.scene`, `.controls`, `.canvas-host`, theme variables,
and class-color variables; delete CSS that only scene-specific files in
`cnn-deepdive` referenced).

**Acceptance:** open `unet-deepdive/index.html`, all 15 scenes are reachable
via dot-pager and arrow keys. Theme toggle works. No console errors.
No content yet — placeholder text "Scene N not yet implemented" is fine.

**Size:** ~2 hours of edits, mostly mechanical. **Should ultrathink:** no.

---

### Agent B — Precompute (depends on: nothing)

**Scope:** all of §3. This is a multi-step pipeline, with clear chunks
that can be sub-agented.

**Sub-agent breakdown** (B fan-out, dispatched by Agent B itself):

- **B1 — Reuse existing artifacts.** Copy `scene64_data.npz` and
  `segmenter.pt` from `cnn-deepdive/precompute/artifacts/` into
  `unet-deepdive/precompute/artifacts/`. Re-run `export_to_js.py`
  (adapted from cnn-deepdive's) to produce `unet_intermediates.npz` and
  `bottleneck_rfields.json` covering 6-12 representative test samples
  with all intermediate activations.
- **B2 — Upsampling demos.** Write `upsample_demos.py`. Take a 4×4 input
  with a few bright cells; produce nearest-neighbor 8×8, bilinear 8×8,
  and a few transposed-conv 8×8 outputs (with hand-picked filters: a
  Gaussian, a "+", an edge filter). Emit `upsample_demos.json`.
- **B3 — Deconv intuition data.** Write `deconv_intuition.py`. For a
  3×3 input and a 3×3 filter, produce the per-input-cell stamp traces
  (one 5×5 frame per stamp) and the corresponding zero-insert + conv
  trace. Plus the matrix-form data for the 1D advanced inset. Emit
  `deconv_traces.json`.
- **B4 — Skip ablation.** Write `skip_ablation.py`. Build a no-skip
  variant of TinyUNet (the decoder takes only the upsampled tensor,
  no concatenation), train on the same data with the same recipe,
  export predictions on the same 6-12 samples used for the with-skip
  comparison. Acceptance: the no-skip model should reach ~0.75-0.85
  pixel accuracy (visibly worse than 0.93+ with skips). Emit
  `noskip_segmenter.pt` and `noskip_predictions.npz`.
- **B5 — Training traces.** Write `training_traces.py`. Reproduce the
  `train_segmenter.py` training loop, but every K steps snapshot
  (a) current loss, (b) current pixel accuracy, (c) the prediction map
  on one fixed test sample. Emit `training_traces.npz` with ~30 frames.
- **B6 — Confusion matrix and failure picks.** Compute the 5×5 confusion
  matrix on the 100-sample test set. Identify the 3 best and 5 worst
  test samples by per-sample accuracy. Emit `confusion_matrix.json`
  and `failure_picks.json`.
- **B7 — Bundling.** Write `data/datasets.js` aggregating all of the
  above into a single `window.DATA` object. Match the conventions of
  `cnn-deepdive/data/datasets.js`.

**Acceptance:** `data/datasets.js` loads in the browser without errors;
`window.DATA` has all keys listed in §3; the file size is reasonable
(<5 MB; if larger, switch large arrays to base64-encoded
Float32Arrays).

**Size:** large, ~6-10 hours total work. **Should ultrathink:** yes,
especially for B3 (deconv data must be carefully designed for the
stamp animation). B should fan out B1-B7 to sub-agents in parallel.

---

### Agent C — Dataset & task scenes (depends on: A, B1, B7)

**Scope:** scenes 1, 2.
**Size:** ~3-4 hours. **Should ultrathink:** no.

---

### Agent D — Encoder, bottleneck, decoder (depends on: A, B1, B7)

**Scope:** scenes 4, 5, 8.
**Size:** ~5-6 hours. Reuses much of `cnn-deepdive/scene9.js`'s painters.
**Should ultrathink:** no.

---

### Agent E — Upsampling deep-dive (depends on: A, B2, B7)

**Scope:** scene 6.
**Size:** ~3-4 hours. **Should ultrathink:** lightly — get the
"learnable vs not" framing right and the cost table accurate.

---

### Agent F — Transposed convolution (depends on: A, B3, B7)

**Scope:** scene 7. The conceptual centerpiece. ~400-500 lines of JS,
plus its own CSS file.

**Sub-agent breakdown** (F fan-out):

- **F1 — Stamp animation.** The core animation: input cells glowing in
  sequence, each one stamping the filter into the output canvas, sums
  accumulating with a running counter. Hover any output cell to see
  which input cells contributed.
- **F2 — Zero-insert view.** The toggle that swaps the stamp animation
  for the equivalent "insert zeros, then run a regular conv" animation.
  A hard-stop visual confirmation: "same numbers, both ways".
- **F3 — Matrix inset (advanced).** Collapsed by default. When opened,
  shows a 1D `4 → 7` example with the 7×4 `W^T` matrix rendered via
  KaTeX, with one row highlighted at a time as the matmul plays out.
- **F4 — Knobs (stride, filter pattern).** The two interactive controls
  at the bottom of the scene; switching a knob re-runs F1 from scratch
  with the new parameters.

**Acceptance:** the viewer can correctly answer the question "if I change
this filter cell from 0 to 1, what happens to the output map?" by
working it out themselves, and the visualization confirms.

**Size:** large, ~10-12 hours. **Should ultrathink:** yes. F should
dispatch F1-F4 to sub-agents and itself act as integrator.

---

### Agent G — Skip connections (depends on: A, B1, B4, B7)

**Scope:** scenes 9, 10.

**Sub-agent breakdown** (G fan-out):

- **G1 — Scene 9: the no-skip vs. with-skip side-by-side.** Wire up the
  four-panel display and the diff overlay. Implement the receptive-field
  hover.
- **G2 — Scene 10: concat vs. residual disambiguation.** Implement the
  two-diagram comparison and the live `α/β` slider demo.

**Acceptance:** the pedagogical contrast in scene 9 is *immediately
visible*: no caption needed to tell the viewer which one is better.

**Size:** large, ~8-10 hours. **Should ultrathink:** yes. G should
fan out G1, G2.

---

### Agent H — Architecture + training (depends on: A, B1, B5, B7)

**Scope:** scenes 11, 12, 13.
**Size:** medium-large, ~6-8 hours. **Should ultrathink:** lightly,
mainly for the training-dynamics scene which has subtle interaction
design choices (scrubber + curves + prediction snapshots all linked).

---

### Agent I — Overview + failure modes (depends on: A, B1, B6, B7)

**Scope:** scenes 0, 14. Scene 0 is the overture; scene 14 is the
honest failure-modes coda.
**Size:** ~3-4 hours. **Should ultrathink:** no.

---

### Agent J — Polish + integration (depends on: C, D, E, F, G, H, I)

**Scope:** end-of-pipeline cleanup. Specifically:

- Walk through all 15 scenes manually in light and dark theme.
  Fix CSS clashes, broken layouts on narrow viewports, missing alt-text.
- Verify `?scene=N&run` works for every scene that supports `&run`.
- Verify keyboard navigation (arrows, dot-pager).
- Verify no console errors or warnings on any scene.
- Compress oversized images and bundle the precompute outputs to keep
  total page weight under a sensible threshold (target <8 MB).
- Write `README.md` (one paragraph: what this is, how to view it,
  how to regenerate the precompute).
- Run `precompute/*.py` end-to-end on a clean checkout to confirm
  reproducibility.
- Smoke-test on a second browser (Firefox).

**Size:** ~4-6 hours. **Should ultrathink:** no.

---

## 5. Style and pedagogy guardrails

Lifted from `cnn-deepdive`'s tone, applied here.

- **Captions are short, grounded, and don't lecture.** They name what is
  on screen and what the viewer should take from it. They never re-state
  what the viewer just clicked through.
- **Click-step engine, not scroll.** Every scene has discrete steps the
  viewer advances through. No magical scroll-driven motion.
- **No surprise math.** KaTeX formulas only appear after they have been
  built up visually. The math labels the picture, never the other way
  round.
- **Hover is for inspection, click is for advancement.** Hover always
  reveals the *meaning* of a cell or arrow; the viewer never has to
  click to learn.
- **Theme-pure colors.** All semantic colors (sky, grass, sun, tree,
  person, ink, accent) come from CSS variables. No `#hex` literals in JS
  except for `parseHex` debug fallbacks.
- **The U-shape diagram is the through-line.** It appears in scene 0,
  scene 8 (without skips), scene 9 (skips highlighted), and scene 11
  (fully labeled). Each appearance adds one more layer of annotation;
  the viewer should feel like they are watching the same picture grow
  up.
- **Failure honesty.** Scene 14 is non-negotiable. A learner who walks
  away thinking "U-Nets are perfect" is poorly served.

---

## 6. Acceptance criteria for the whole deepdive

The deepdive is done when:

1. All 15 scenes render in light and dark theme without console errors.
2. A first-time viewer with the CNN deepdive background can:
   - Explain the difference between classification and segmentation outputs.
   - Draw the U-shape from memory and label every edge with shape and op.
   - Describe in one sentence what a transposed convolution does, using
     either the stamp or zero-insert intuition.
   - Explain *why* the skip connection exists, in terms of "what got
     lost in the bottleneck".
   - Distinguish a U-Net concat skip from a ResNet residual add and
     name a reason each one exists.
   - Identify a class of input the model will fail on and say why.
3. Every precompute script runs end-to-end on a clean machine with a
   single `make` or `bash run_precompute.sh`.
4. Total page weight under 8 MB.
5. Time-to-interactive on first scene under 1.5 s on a mid-tier laptop.

---

## 7. Resolved decisions

The human resolved all open questions on dispatch:

1. **Retrain the U-Net** under a fixed random seed in
   `unet-deepdive/precompute/`. Do **not** copy `segmenter.pt` or
   `scene64_data.npz` from `cnn-deepdive/precompute/artifacts/`. Each
   precompute script in this folder must reproduce its inputs from
   scratch. Use seed `43` (matching the cnn-deepdive convention) so
   prediction maps remain consistent across reruns.
2. **15 scenes is final.** Do not trim.
3. **Scene 7 advanced matrix inset: keep**, collapsed behind a
   "show me the matrix" affordance.
4. **Scene 13 weight-magnitude sparkline: keep.**
5. **Agents A and B run in parallel** as the first wave.
