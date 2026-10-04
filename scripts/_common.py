"""Path setup shared by the scripts (lets them import spectrum_intel from the project root)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
BASELINE_MODEL = MODELS / "cnn_baseline.pt"
DOPPLER_MODEL = MODELS / "cnn_doppler.pt"
METRICS = RESULTS / "metrics.json"


def update_metrics(section: str, values: dict):
    """Merge one section into results/metrics.json."""
    RESULTS.mkdir(exist_ok=True)
    data = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    data[section] = values
    METRICS.write_text(json.dumps(data, indent=2))
