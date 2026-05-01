"""Top-level orchestrator: builds ``data/datasets.js`` from scratch.

Runs each stage in order, asserting invariants per the brief:
  1. Train shapelets classifier  (assert test_acc >= 0.95)
  2. Train scene64 U-Net          (assert mean pixel acc >= 0.90)
  3. Activation maximization     (assert all AM variances > 1e-4)
  4. Receptive field metadata    (analytic)
  5. Export to JS                (file size <= 4 MB target)
  6. Smoke-test parse with node

Usage:
    python3 build_data.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))


def run(cmd, cwd=HERE):
    print(f'\n=== {cmd} ===')
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f'FAILED: {cmd}')
        sys.exit(res.returncode)


def main():
    run('python3 train_shapelets.py')
    run('python3 train_segmenter.py')
    run('python3 activation_max.py')
    run('python3 receptive_fields.py')
    run('python3 export_to_js.py')

    # Final node parse-check.
    out = os.path.join(ROOT, 'data', 'datasets.js')
    code = (
        "globalThis.window={}; "
        "eval(require('fs').readFileSync('" + out + "','utf8')); "
        "if(!window.DATA||!window.DATA.shapelets){throw new Error('missing DATA');} "
        "console.log('OK keys:', Object.keys(window.DATA).join(','), "
        "'  shapelets.testAccuracy:', window.DATA.shapelets.testAccuracy, "
        "'  scene64.meanPixelAccuracy:', window.DATA.scene64.meanPixelAccuracy);"
    )
    run(f'node -e "{code}"', cwd=HERE)
    size = os.path.getsize(out)
    print(f'\nFinal datasets.js size: {size:,} bytes')


if __name__ == '__main__':
    main()
