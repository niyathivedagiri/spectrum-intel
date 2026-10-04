"""Run every experiment in order and refresh results/.

    python scripts/run_all.py              # uses the trained models in models/
    python scripts/run_all.py --retrain    # retrains both CNNs first (~20-30 min on a laptop)
"""
import argparse
import subprocess
import sys

from _common import BASELINE_MODEL, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--retrain", action="store_true")
a = ap.parse_args()

steps = [["scripts/train_classifier.py"] if a.retrain or not BASELINE_MODEL.exists()
         else ["scripts/train_classifier.py", "--evaluate-only"]]
steps += [["scripts/preview_signals.py"], ["scripts/evaluate_detection.py"], ["scripts/run_spectrum_demo.py"],
          ["scripts/run_leo_experiment.py"] + (["--retrain"] if a.retrain else [])]

for cmd in steps:
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)
print("\nAll done. Figures and numbers are in results/.")
