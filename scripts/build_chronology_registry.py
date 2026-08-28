from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.chronology.registry import ingest
if __name__ == "__main__": print(ingest(Path("data/chronology_sources"), Path("data/chronology_registry/v1")))
