
import argparse
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from normalize import norm
from common import PASSAGES_PATH, BM25_INDEX_PATH, load_jsonl

DATA_PATH = str(PASSAGES_PATH)
INDEX_PATH = str(BM25_INDEX_PATH)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)



def load_passages(path: str) -> list[dict]:
    rows = load_jsonl(path)
    rows.sort(key=lambda r: (r["page_id"], r["position"]))
    return rows



def tokenize(text: str) -> list[str]:
    t = norm(text)
    t = _PUNCT_RE.sub(" ", t)
    return t.split()



class BM25Index:
    """Wraps rank_bm25.BM25Okapi with ids, metadata, and save/load."""

    def __init__(self, ids, raw_texts, metadatas, bm25: BM25Okapi):
        self.ids = ids                  # list[str], row order == bm25 corpus order
        self.raw_texts = raw_texts      # list[str], for display / debugging
        self.metadatas = metadatas      # list[dict]
        self.bm25 = bm25


    @classmethod
    def build(cls, data_path: str = DATA_PATH) -> "BM25Index":
        rows = load_passages(data_path)
        if not rows:
            raise SystemExit(f"No passages found in {data_path}")

        ids = [r["passage_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit("Duplicate passage_id values in passages.jsonl")

        raw_texts = [r["text"] for r in rows]
        metadatas = [{k: v for k, v in r.items() if k != "text"} for r in rows]

        tokenized = [tokenize(t) for t in raw_texts]
        empty = [pid for pid, toks in zip(ids, tokenized) if not toks]
        if empty:
            raise SystemExit(f"Empty token list after normalization for: {empty}")

        bm25 = BM25Okapi(tokenized)
        print(f"Built BM25 over {len(ids)} passages.")
        return cls(ids, raw_texts, metadatas, bm25)


    def save(self, path: str = INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "raw_texts": self.raw_texts,
                    "metadatas": self.metadatas,
                    "bm25": self.bm25,
                },
                f,
            )
        print(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: str = INDEX_PATH) -> "BM25Index":
        if not Path(path).exists():
            raise SystemExit(f"No index at {path}. Run without --skip-build first.")
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["ids"], d["raw_texts"], d["metadatas"], d["bm25"])


    def search(self, query: str, k: int = 10, where: dict | None = None):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        if where:
            order = [
                i for i in order
                if all(self.metadatas[i].get(key) == val for key, val in where.items())
            ]

        return [(self.ids[i], float(scores[i])) for i in order[:k]]


def main():
    parser = argparse.ArgumentParser(description="Build and save the BM25 passage index.")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    args = parser.parse_args()
    BM25Index.build(args.data).save(args.index)


if __name__ == "__main__":
    main()


