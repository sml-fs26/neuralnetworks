#!/usr/bin/env bash
# Run every precompute step from a clean checkout, in dependency order.
#
# B1: train the with-skip TinyUNet (gates B4..B7)
# B2: upsample demos                            (independent)
# B3: deconv intuition data                     (independent)
# B4: train no-skip ablation                    (depends on scene64 data)
# B5: re-train with traces                      (depends on scene64 data)
# B6: confusion matrix and failure picks        (depends on B1)
# B7: aggregate everything into data/datasets.js
set -euo pipefail
cd "$(dirname "$0")"

echo "=== B1: train with-skip TinyUNet ==="
python3 train_segmenter.py

echo "=== B2: upsample demos ==="
python3 upsample_demos.py

echo "=== B3: deconv intuition data ==="
python3 deconv_intuition.py

echo "=== B4: train no-skip U-Net ==="
python3 skip_ablation.py

echo "=== B5: training traces ==="
python3 training_traces.py

echo "=== B6: confusion matrix and failure picks ==="
python3 confusion_and_picks.py

echo "=== B7: export to data/datasets.js ==="
python3 export_to_js.py

echo "All precompute steps done."
