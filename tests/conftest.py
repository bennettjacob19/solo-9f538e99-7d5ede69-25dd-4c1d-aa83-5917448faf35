import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# App modules import each other flat (``import lrmath``), exactly as they do
# inside the container where ``app/`` is copied to ``/app``.
sys.path.insert(0, str(ROOT / "app"))
