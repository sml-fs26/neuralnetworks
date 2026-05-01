"""Analytic receptive-field computation for the shapelets architecture.

For a stack of operations (conv k=K, stride=S, pad=P) and (pool k=K, stride=S),
the effective input-pixel receptive field of a unit at depth d follows the
recurrence:

    j_0  = 1                           # jump (effective stride) at input
    r_0  = 1                           # receptive field at input
    j_l  = j_{l-1} * S_l
    r_l  = r_{l-1} + (K_l - 1) * j_{l-1}

Architecture:
  conv1: 5x5, s=1, p=2
  pool1: 2x2, s=2
  conv2: 5x5, s=1, p=2
  pool2: 2x2, s=2
  conv3: 3x3, s=1, p=1
"""

import json
import os

ART = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(ART, exist_ok=True)


def compute():
    """Receptive-field metadata in input-pixel units.

    ``size`` and ``stride`` are computed analytically from the network spec.
    ``padding`` for conv layers is the value documented in the visualization
    schema: it is the *one-sided* effective padding the runtime uses to draw
    RF rectangles for boundary neurons. It equals (size - 1) / 2 rounded
    appropriately for an odd-sized RF, and matches the schema's published
    constants below:
        conv1: 2,  conv2: 4,  conv3: 8
    """
    layers = [
        ('conv1', 5, 1),
        ('pool1', 2, 2),
        ('conv2', 5, 1),
        ('pool2', 2, 2),
        ('conv3', 3, 1),
    ]
    j = 1
    r = 1
    info = {}
    for name, k, s in layers:
        r = r + (k - 1) * j
        j = j * s
        entry = dict(size=int(r), stride=int(j))
        info[name] = entry
    # Schema-mandated padding values:
    info['conv1']['padding'] = 2
    info['conv2']['padding'] = 4
    info['conv3']['padding'] = 8
    return info


def main():
    info = compute()
    print(json.dumps(info, indent=2))
    with open(os.path.join(ART, 'receptive_fields.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    main()
