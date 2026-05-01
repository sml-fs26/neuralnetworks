"""Activation-maximization images and top-9 maximally-activating training
images per visualized neuron.

Visualizes:
  - 8 channels of conv1
  - 8 channels of conv2
  - 4 channels of conv3

Uses gradient ascent on the input image with TV + L2 regularization.
Asserts every AM image has variance > 1e-4 (not constant).
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from train_shapelets import ShapeletsCNN

ART = os.path.join(os.path.dirname(__file__), 'artifacts')


def set_seed(s):
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def total_variation(x):
    # x: (1,1,H,W)
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dh.abs().mean() + dw.abs().mean())


def _pre_relu_features(model, x):
    """Forward pass exposing pre-ReLU conv outputs.

    Returns dict with conv1, conv2, conv3 = pre-ReLU activations.
    """
    c1_pre = model.conv1(x)
    c1 = torch.relu(c1_pre)
    p1 = torch.max_pool2d(c1, 2)
    c2_pre = model.conv2(p1)
    c2 = torch.relu(c2_pre)
    p2 = torch.max_pool2d(c2, 2)
    c3_pre = model.conv3(p2)
    return dict(conv1=c1_pre, conv2=c2_pre, conv3=c3_pre)


def _objective(feat, mode='center'):
    """Compute a scalar to maximize from a (H, W) feature map.

    'center'  - value at the spatial center (best for low-level conv1, gives
                a localized pattern centered in the image, exactly the kernel
                shape).
    'mean'    - mean across spatial (uniform input gradient for linear layers,
                BAD for conv1 — produces saturated flat images).
    'topk'    - mean of the top-K spatial activations (good middle ground).
    """
    H, W = feat.shape[-2:]
    if mode == 'center':
        return feat[..., H // 2, W // 2]
    if mode == 'topk':
        k = max(1, (H * W) // 8)
        flat = feat.reshape(-1)
        topk, _ = torch.topk(flat, k)
        return topk.mean()
    return feat.mean()


def activation_max(model, layer_name, channel, steps=256, lr=0.1,
                   tv_weight=1e-3, l2_weight=1e-4, seed=0,
                   init_scale=0.2, jitter_pad=2,
                   objective='center'):
    """Gradient ascent on input to maximize the pre-ReLU activation of
    (layer_name, channel). For linear conv layers the *mean* over spatial
    has spatially-uniform input gradient and yields saturated flat images;
    we instead maximize a *spatially localized* objective by default.
    """
    set_seed(seed)
    x = torch.randn(1, 1, 28, 28) * init_scale + 0.5
    x = x.clamp(0, 1).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        if jitter_pad > 0:
            sh = torch.randint(-jitter_pad, jitter_pad + 1, (2,)).tolist()
            xj = torch.roll(x, shifts=sh, dims=(2, 3))
        else:
            xj = x
        feats = _pre_relu_features(model, xj)
        feat = feats[layer_name][0, channel]
        act = _objective(feat, objective)
        reg = tv_weight * total_variation(x) + l2_weight * (x ** 2).mean()
        loss = -act + reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.data.clamp_(0, 1)
    return x.detach().squeeze().numpy().astype(np.float32)


def activation_max_signed(model, layer_name, channel, steps=400, step_size=0.02,
                          seed=0, init_scale=0.2, objective='center'):
    """Signed-gradient ascent."""
    set_seed(seed)
    x = torch.randn(1, 1, 28, 28) * init_scale + 0.5
    x = x.clamp(0, 1).clone().detach().requires_grad_(True)
    for step in range(steps):
        if x.grad is not None:
            x.grad.zero_()
        feats = _pre_relu_features(model, x)
        feat = feats[layer_name][0, channel]
        act = _objective(feat, objective)
        act.backward()
        with torch.no_grad():
            x.data = (x.data + step_size * x.grad.sign()).clamp_(0, 1)
    return x.detach().squeeze().numpy().astype(np.float32)


def top9_for_neuron(model, layer_name, channel, X_t):
    """Compute mean activation of (layer_name, channel) over each image in
    X_t and return the indices of the top-9 activators."""
    model.eval()
    bs = 256
    n = X_t.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = X_t[i:i + bs]
            _, inter = model(xb, return_intermediates=True)
            feat = inter[layer_name][:, channel]
            s = feat.mean(dim=(1, 2)).cpu().numpy()
            scores[i:i + bs] = s
    order = np.argsort(-scores)
    return order[:9].tolist(), scores


def main():
    ckpt = torch.load(os.path.join(ART, 'shapelets.pt'), map_location='cpu')
    model = ShapeletsCNN()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    data = np.load(os.path.join(ART, 'shapelets_data.npz'))
    Xtr = data['Xtr']
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1).float()

    spec = (
        [('conv1', c) for c in range(8)] +
        [('conv2', c) for c in range(8)] +
        [('conv3', c) for c in range(4)]
    )

    neurons = []
    am_variances = []

    # Try multiple hyperparameter sets if a result is too flat. Mixes Adam
    # with regularization and FGSM-style signed-gradient ascent. The default
    # objective is 'center' (single-pixel value at the center of the feature
    # map), which yields spatially structured patterns even for linear conv1.
    hp_attempts = [
        ('adam', dict(lr=0.1, tv_weight=1e-3, l2_weight=1e-4, steps=256,
                      init_scale=0.2, jitter_pad=2, objective='center')),
        ('adam', dict(lr=0.2, tv_weight=5e-4, l2_weight=1e-4, steps=400,
                      init_scale=0.3, jitter_pad=3, objective='center')),
        ('signed', dict(step_size=0.03, steps=400, init_scale=0.2, objective='center')),
        ('adam', dict(lr=0.3, tv_weight=2e-4, l2_weight=0.0, steps=400,
                      init_scale=0.4, jitter_pad=4, objective='topk')),
        ('signed', dict(step_size=0.02, steps=600, init_scale=0.3, objective='topk')),
    ]

    for idx, (layer, ch) in enumerate(spec):
        img = None
        best_var = -1.0
        for hp_i, (method, hp) in enumerate(hp_attempts):
            for seed in range(3):
                if method == 'adam':
                    cand = activation_max(model, layer, ch,
                                          seed=100 + idx * 13 + hp_i * 7 + seed * 11,
                                          **hp)
                else:  # signed
                    cand = activation_max_signed(model, layer, ch,
                                                 seed=100 + idx * 13 + hp_i * 7 + seed * 11,
                                                 **hp)
                v = float(cand.var())
                if v > best_var:
                    best_var = v
                    img = cand
                if best_var > 1e-3:
                    break
            if best_var > 1e-3:
                break
        var = float(img.var())
        am_variances.append(var)
        top9, _ = top9_for_neuron(model, layer, ch, Xtr_t)
        neurons.append(dict(
            layer=layer, channel=int(ch),
            image=img.tolist(),
            top9Indices=[int(i) for i in top9],
        ))
        print(f'  [{idx+1:02d}/{len(spec)}] {layer} ch{ch}: var={var:.5f} top9={top9[:3]}...')

    am_variances = np.array(am_variances)
    print(f'\nAM variance: min={am_variances.min():.5f}  max={am_variances.max():.5f}')
    assert (am_variances > 1e-4).all(), \
        f'Some AM images have variance <= 1e-4: {am_variances}'

    # Persist
    np.savez(
        os.path.join(ART, 'am.npz'),
        variances=am_variances,
        layers=np.array([n['layer'] for n in neurons]),
        channels=np.array([n['channel'] for n in neurons]),
        images=np.stack([np.array(n['image'], dtype=np.float32) for n in neurons]),
        top9=np.stack([np.array(n['top9Indices'], dtype=np.int64) for n in neurons]),
    )

    # Also pre-compute top-9 for conv2 neurons (the schema asks for shapelets.conv2Top9
    # on all 16 conv2 channels, not just the 8 we visualize).
    conv2_top9_all = []
    for ch in range(16):
        t9, _ = top9_for_neuron(model, 'conv2', ch, Xtr_t)
        conv2_top9_all.append([int(i) for i in t9])
    np.savez(
        os.path.join(ART, 'conv2_top9.npz'),
        top9=np.array(conv2_top9_all, dtype=np.int64),
    )

    print('Activation maximization done.')


if __name__ == '__main__':
    main()
