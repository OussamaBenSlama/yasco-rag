import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PASSAGES_PATH = DATA_DIR / "passages.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
BM25_INDEX_PATH = ROOT / "bm25_index.pkl"
CHROMA_DB_DIR = ROOT / "chroma_db"
CHROMA_COLLECTION = "passages"


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
