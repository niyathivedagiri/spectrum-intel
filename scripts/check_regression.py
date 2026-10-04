"""Check that the original validated results have not changed.

    python scripts/check_regression.py            # re-run the original pipeline, then compare (~3 min)
    python scripts/check_regression.py --no-run   # only compare the current results/metrics.json

Compares results/metrics.json with the frozen copy results/validated/metrics_v1.json.
Every number must match within --tol (default 0.005, i.e. half a percentage point,
which allows for tiny floating-point differences between machines/library versions).
"""
import argparse
import json
import subprocess
import sys

from _common import METRICS, RESULTS, ROOT

FROZEN = RESULTS / "validated" / "metrics_v1.json"


def compare(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing key", None))
            else:
                compare(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}", None))
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        out.append((path, "number", abs(float(a) - float(b))))
    elif a != b:
        out.append((path, f"{a!r} vs {b!r}", None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    if not a.no_run:
        subprocess.run([sys.executable, "scripts/run_all.py"], cwd=ROOT, check=True)
    frozen, now = json.loads(FROZEN.read_text()), json.loads(METRICS.read_text())
    diffs = compare(frozen, now)
    bad = [d for d in diffs if d[2] is None or d[2] > a.tol]
    worst = max((d[2] for d in diffs if d[2] is not None), default=0.0)
    for section in frozen:
        n = sum(1 for d in diffs if d[0].startswith(f"/{section}"))
        nb = sum(1 for d in bad if d[0].startswith(f"/{section}"))
        print(f"  {section:14s} {n:5d} values checked, {nb} outside tolerance")
    print(f"Largest numeric difference: {worst:.3g} (tolerance {a.tol})")
    if bad:
        for p, what, d in bad[:20]:
            print(f"  CHANGED {p}: {what} {'' if d is None else f'diff {d:.3g}'}")
        sys.exit(1)
    print("REGRESSION CHECK PASSED: validated results unchanged.")


if __name__ == "__main__":
    main()
