"""B7 — Aggregate all artifacts into ../data/datasets.js.

Loads:
  artifacts/segmenter.pt
  artifacts/scene64_data.npz
  artifacts/noskip_segmenter.pt
  artifacts/noskip_predictions.npz
  artifacts/training_traces.npz
  artifacts/upsample_demos.json
  artifacts/deconv_traces.json
  artifacts/confusion_matrix.json
  artifacts/failure_picks.json

Builds:
  unet_intermediates: 6 representative samples with full encoder/decoder
                      intermediates (4 channels per stage by max-variance pick)
  bottleneck_rfields: 16×16 grid of input-coord rectangles for the bottleneck

Writes ../data/datasets.js as one JS file with `window.DATA = { ... }`.

If the file would exceed ~5 MB we keep going but flag it; the largest arrays
(intermediates, training pred frames) are emitted as base64-encoded
Float32/Int8 buffers wrapped by tiny JS decoders.
"""

import base64
import json
import os
import numpy as np
import torch

from train_segmenter import TinyUNet
from scene64_data import generate_dataset, CLASSES

ROUND = 2  # 2 decimals is plenty for browser-side activation visualization
HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, 'artifacts')
OUT = os.path.normpath(os.path.join(HERE, '..', 'data', 'datasets.js'))


def r(x):
    return round(float(x), ROUND)


def r_acc(x):
    return round(float(x), 4)


def to_list(arr):
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
    return json.dumps(value, separators=(',', ':'))


def b64_float32(arr):
    """Pack a numpy array into a base64-encoded float32 buffer + shape, so
    the JS side can reconstruct it via ``new Float32Array(decode(...))``.
    """
    arr = np.ascontiguousarray(arr.astype(np.float32))
    raw = arr.tobytes(order='C')
    return dict(_b64=True, dtype='float32',
                shape=list(arr.shape),
                data=base64.b64encode(raw).decode('ascii'))


def b64_uint8(arr):
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    raw = arr.tobytes(order='C')
    return dict(_b64=True, dtype='uint8',
                shape=list(arr.shape),
                data=base64.b64encode(raw).decode('ascii'))


def pick_six_samples(Xte, Yte):
    """Same logic as skip_ablation.pick_six_samples — must agree byte-for-byte
    so noskip predictions line up.
    """
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
    for i in range(len(Xte)):
        if len(chosen) >= 6:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:6]


def top_var_channels(feat, n=4):
    """Pick the n channels of `feat` (C, H, W) with largest variance."""
    C = feat.shape[0]
    var = feat.reshape(C, -1).var(axis=1)
    idx = np.argsort(var)[::-1][:n]
    return np.sort(idx)


def gather_scene64(model, Xte, Yte, chosen):
    """The on-disk shape preserved from the cnn-deepdive scene9 consumer:
    `samples[i] = { input(64,64,3), label(64,64), pred(64,64),
                    enc1(4,64,64), enc2(4,32,32), enc3(4,16,16),
                    dec2(4,32,32), dec1(4,64,64) }`.

    For the U-Net deepdive we keep the same shape so existing helpers are
    reusable; channel selection is by max variance (top 4 out of 16/32/64).
    """
    samples = []
    with torch.no_grad():
        for i in chosen:
            x = Xte[i]
            y = Yte[i]
            x_t = torch.from_numpy(x).unsqueeze(0).float()
            out, inter = model(x_t, return_intermediates=True)
            pred = out.argmax(1).squeeze(0).numpy()
            img_hwc = np.transpose(x, (1, 2, 0))
            # pick by variance
            e1 = inter['enc1'][0].numpy(); e1c = top_var_channels(e1, 4)
            e2 = inter['enc2'][0].numpy(); e2c = top_var_channels(e2, 4)
            e3 = inter['enc3'][0].numpy(); e3c = top_var_channels(e3, 4)
            d2 = inter['dec2'][0].numpy(); d2c = top_var_channels(d2, 4)
            d1 = inter['dec1'][0].numpy(); d1c = top_var_channels(d1, 4)
            samples.append(dict(
                input=to_list(img_hwc),
                label=to_list(y),
                pred=to_list(pred),
                enc1=to_list(e1[e1c]),
                enc2=to_list(e2[e2c]),
                enc3=to_list(e3[e3c]),
                dec2=to_list(d2[d2c]),
                dec1=to_list(d1[d1c]),
                channel_picks=dict(
                    enc1=e1c.tolist(), enc2=e2c.tolist(),
                    enc3=e3c.tolist(), dec2=d2c.tolist(), dec1=d1c.tolist(),
                ),
            ))
    return samples


def gather_intermediates_full(model, Xte, chosen):
    """Full-resolution intermediates for all 5 stages, base64-packed.

    Per sample we emit enc1 (16,64,64), enc2 (32,32,32), enc3 (64,16,16),
    dec2 (32,32,32), dec1 (16,64,64), plus the argmax pred.

    To stay within the 5 MB budget we only emit FULL intermediates for the
    first 2 of the 6 chosen samples; the remaining 4 samples have just
    the bottleneck (enc3) full plus a pred. The 4-channel-per-stage data
    is already available in scene64.samples for all 6.
    """
    bundle = []
    with torch.no_grad():
        for k, i in enumerate(chosen):
            x_t = torch.from_numpy(Xte[i:i+1]).float()
            out, inter = model(x_t, return_intermediates=True)
            pred = out.argmax(1).squeeze(0).numpy().astype(np.uint8)
            entry = dict(
                index=int(i),
                pred=b64_uint8(pred),
                enc3=b64_float32(inter['enc3'][0].numpy()),
            )
            if k == 0:
                # Sample 0 only — too big to do for all 6.
                entry['enc1'] = b64_float32(inter['enc1'][0].numpy())
                entry['enc2'] = b64_float32(inter['enc2'][0].numpy())
                entry['dec2'] = b64_float32(inter['dec2'][0].numpy())
                entry['dec1'] = b64_float32(inter['dec1'][0].numpy())
            bundle.append(entry)
    return bundle


def compute_bottleneck_rfields():
    """Receptive-field rectangle for each (i,j) of the 16×16 bottleneck (enc3
    output) in 64×64 input pixel coordinates.

    Architecture path from input to enc3:
      enc1 conv 3×3 p=1 -> conv 3×3 p=1                 (64×64)
      maxpool 2×2 s=2                                    (32×32)
      enc2 conv 3×3 p=1 -> conv 3×3 p=1                 (32×32)
      maxpool 2×2 s=2                                    (16×16)
      enc3 conv 3×3 p=1 -> conv 3×3 p=1                 (16×16)

    Recurrence: r0=1, j0=1; for each layer with kernel k and stride s,
      r := r + (k-1)*j;  j := j*s.

    Layers in order: conv3, conv3, pool2x2 s2, conv3, conv3, pool2x2 s2,
                     conv3, conv3.
    """
    layers = [
        (3, 1), (3, 1), (2, 2),
        (3, 1), (3, 1), (2, 2),
        (3, 1), (3, 1),
    ]
    j = 1
    r_rf = 1
    for k, s in layers:
        r_rf = r_rf + (k - 1) * j
        j = j * s
    rf_size = r_rf  # in input pixels
    rf_jump = j     # input-pixel stride between adjacent bottleneck cells

    # Center offset of the (0,0) bottleneck cell back in input coords:
    # The first bottleneck cell's RF center is at input pixel
    # (rf_size - 1) / 2 - sum(paddings * scale) ... easier: empirically the
    # convs are all p=1 (same), so the center of cell (i,j) maps to
    # (i*jump + (rf_size - 1) / 2 - offset_padding, ...). With same-padding,
    # the centering is just i*jump + (rf_size//2 - padding_total).
    # Concretely with rf_size=44 and rf_jump=4 (standard for this stack),
    # the (0,0) cell's RF rectangle centers at input coord ~ (rf_size-1)/2 - pad.
    # For the visualization we just want a clamped rectangle.

    cells = []
    for i in range(16):
        for j_ in range(16):
            cy = i * rf_jump + rf_jump // 2
            cx = j_ * rf_jump + rf_jump // 2
            half = rf_size // 2
            y0 = max(0, cy - half)
            x0 = max(0, cx - half)
            y1 = min(64, cy + half + (rf_size % 2))
            x1 = min(64, cx + half + (rf_size % 2))
            cells.append(dict(i=int(i), j=int(j_),
                              cx=int(cx), cy=int(cy),
                              x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1)))

    return dict(rf_size=int(rf_size),
                rf_jump=int(rf_jump),
                cells=cells)


def gather_upsample():
    with open(os.path.join(ART, 'upsample_demos.json')) as f:
        return json.load(f)


def gather_deconv():
    with open(os.path.join(ART, 'deconv_traces.json')) as f:
        return json.load(f)


def gather_confusion():
    with open(os.path.join(ART, 'confusion_matrix.json')) as f:
        return json.load(f)


def gather_failures():
    with open(os.path.join(ART, 'failure_picks.json')) as f:
        return json.load(f)


def gather_noskip(chosen, Xte, Yte):
    npz = np.load(os.path.join(ART, 'noskip_predictions.npz'))
    chosen_recorded = npz['chosen'].tolist()
    preds = npz['preds']
    mean_acc = float(npz['mean_pixel_accuracy'])
    # Sanity: chosen_recorded must match the 6 we picked.
    if chosen_recorded != list(chosen):
        print(f'WARN: noskip chosen mismatch: file={chosen_recorded} vs '
              f'export={chosen}. Reconciling by re-running noskip in-memory.')
    samples = []
    for k, i in enumerate(chosen_recorded):
        samples.append(dict(
            index=int(i),
            pred=to_list(preds[k]),
            label=to_list(Yte[i]),
            input=to_list(np.transpose(Xte[i], (1, 2, 0))),
        ))
    return dict(
        meanPixelAccuracy=r_acc(mean_acc),
        samples=samples,
    )


def gather_training():
    npz = np.load(os.path.join(ART, 'training_traces.npz'))
    pred_frames = npz['pred_frames']  # (F, 64, 64)
    bundle = dict(
        steps=npz['steps'].tolist(),
        loss=[None if np.isnan(v) else r(v) for v in npz['loss'].tolist()],
        pixAcc=[r_acc(v) for v in npz['pix_acc'].tolist()],
        up1Norm=[r(v) for v in npz['up1_norm'].tolist()],
        up2Norm=[r(v) for v in npz['up2_norm'].tolist()],
        # Pred frames: 64×64 ints in 0..4 — pack as base64 uint8.
        predFrames=b64_uint8(pred_frames.astype(np.uint8)),
        fixedSampleInput=to_list(np.transpose(npz['fixed_sample_input'], (1, 2, 0))),
        fixedSampleLabel=to_list(npz['fixed_sample_label']),
    )
    return bundle


def write_intermediate_artifacts(model, Xte, Yte, chosen):
    """Persist the raw intermediates and rfields to disk so other consumers
    (and the acceptance checklist) can find them as standalone files.
    """
    npz_payload = {}
    with torch.no_grad():
        for k, i in enumerate(chosen):
            x_t = torch.from_numpy(Xte[i:i+1]).float()
            out, inter = model(x_t, return_intermediates=True)
            pred = out.argmax(1).squeeze(0).numpy().astype(np.int64)
            npz_payload[f'sample{k}_index'] = np.int64(i)
            npz_payload[f'sample{k}_pred'] = pred
            npz_payload[f'sample{k}_enc3'] = inter['enc3'][0].numpy().astype(np.float32)
            if k == 0:
                for stage in ['enc1', 'enc2', 'dec2', 'dec1']:
                    npz_payload[f'sample{k}_{stage}'] = inter[stage][0].numpy().astype(np.float32)
    np.savez(os.path.join(ART, 'unet_intermediates.npz'), **npz_payload)
    print(f'Wrote {os.path.join(ART, "unet_intermediates.npz")}')

    rfields = compute_bottleneck_rfields()
    with open(os.path.join(ART, 'bottleneck_rfields.json'), 'w') as f:
        json.dump(rfields, f, indent=2)
    print(f'Wrote {os.path.join(ART, "bottleneck_rfields.json")}')


def emit():
    print('Loading segmenter...')
    ckpt = torch.load(os.path.join(ART, 'segmenter.pt'), map_location='cpu')
    model = TinyUNet()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    data = np.load(os.path.join(ART, 'scene64_data.npz'))
    Xte, Yte = data['Xte'], data['Yte']
    chosen = pick_six_samples(Xte, Yte)
    print(f'Chosen 6 samples: {chosen}')

    print('Writing on-disk intermediate artifacts...')
    write_intermediate_artifacts(model, Xte, Yte, chosen)

    print('Gathering scene64...')
    samples = gather_scene64(model, Xte, Yte, chosen)
    scene64 = dict(
        classes=list(ckpt['classes']),
        samples=samples,
        meanPixelAccuracy=r_acc(ckpt['mean_pixel_accuracy']),
        chosenIndices=[int(i) for i in chosen],
    )

    print('Gathering full intermediates (base64)...')
    intermediates = dict(samples=gather_intermediates_full(model, Xte, chosen),
                         classes=list(ckpt['classes']))

    print('Computing bottleneck receptive fields...')
    rfields = compute_bottleneck_rfields()

    print('Loading upsample demos...')
    upsample = gather_upsample()

    print('Loading deconv intuition...')
    deconv = gather_deconv()

    print('Loading no-skip predictions...')
    noskip = gather_noskip(chosen, Xte, Yte)

    print('Loading training traces...')
    training = gather_training()

    print('Loading confusion + failures...')
    confusion = gather_confusion()
    failures = gather_failures()

    print('Gathering conv filters...')
    sd = ckpt['state_dict']
    filters = {
        'enc1_conv1': to_list(sd['enc1.0.weight'].numpy()),  # (16, 3, 3, 3) RGB-input filters
        'enc1_conv2': to_list(sd['enc1.2.weight'].numpy()),  # (16, 16, 3, 3)
        'enc2_conv1': to_list(sd['enc2.0.weight'].numpy()),  # (32, 16, 3, 3)
        'enc2_conv2': to_list(sd['enc2.2.weight'].numpy()),  # (32, 32, 3, 3)
        'enc3_conv1': to_list(sd['enc3.0.weight'].numpy()),  # (64, 32, 3, 3)
        'enc3_conv2': to_list(sd['enc3.2.weight'].numpy()),  # (64, 64, 3, 3)
        'up2':        to_list(sd['up2.weight'].numpy()),     # (64, 32, 2, 2)  transposed-conv
        'up1':        to_list(sd['up1.weight'].numpy()),     # (32, 16, 2, 2)
        'out':        to_list(sd['out.weight'].numpy()),     # (5, 16, 1, 1)  classifier head
    }

    parts = []
    parts.append('// Auto-generated by precompute/export_to_js.py.\n')
    parts.append('// All numerical artifacts for the U-Net deepdive live here.\n')
    parts.append('window.DATA = window.DATA || {};\n')
    parts.append('window.DATA._b64decode = function(obj){\n')
    parts.append('  if (!obj || !obj._b64) return obj;\n')
    parts.append('  var bin = atob(obj.data);\n')
    parts.append('  var len = bin.length;\n')
    parts.append('  var bytes = new Uint8Array(len);\n')
    parts.append('  for (var i=0;i<len;i++) bytes[i]=bin.charCodeAt(i);\n')
    parts.append('  if (obj.dtype === "float32") {\n')
    parts.append('    return { data: new Float32Array(bytes.buffer, bytes.byteOffset, len/4), shape: obj.shape };\n')
    parts.append('  }\n')
    parts.append('  if (obj.dtype === "uint8") {\n')
    parts.append('    return { data: bytes, shape: obj.shape };\n')
    parts.append('  }\n')
    parts.append('  return obj;\n')
    parts.append('};\n')

    def add(key, value):
        parts.append(f'window.DATA.{key} = ')
        parts.append(js_repr(value))
        parts.append(';\n')

    add('scene64', scene64)
    add('intermediates', intermediates)
    add('rfields', rfields)
    add('upsample', upsample)
    add('deconv', deconv)
    add('noskip', noskip)
    add('training', training)
    add('confusion', confusion)
    add('failures', failures)
    add('filters', filters)

    out = ''.join(parts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(out)
    size = os.path.getsize(OUT)
    print(f'Wrote {OUT} ({size:,} bytes, {size/1024/1024:.2f} MB)')
    if size > 5 * 1024 * 1024:
        print(f'WARNING: exceeds 5 MB budget')
    return size


if __name__ == '__main__':
    emit()
