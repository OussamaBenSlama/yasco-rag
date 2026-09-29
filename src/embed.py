
import argparse
import json 
import chromadb
from sentence_transformers import SentenceTransformer

from normalize import norm
from common import PASSAGES_PATH, CHROMA_DB_DIR, CHROMA_COLLECTION, load_jsonl

DATA_PATH = str(PASSAGES_PATH)
DB_DIR = str(CHROMA_DB_DIR)
COLLECTION = CHROMA_COLLECTION
MODEL_NAME = "intfloat/multilingual-e5-small"   
BATCH_SIZE = 16
MAX_SEQ_LEN = 512



def load_passages(path: str) -> list[dict]:
    rows = load_jsonl(path)
    rows.sort(key=lambda r: (r["page_id"], r["position"]))
    return rows


def clean_metadata(row: dict) -> dict:
    meta = {}
    for key, val in row.items():
        if key == "text":
            continue
        if val is None:
            meta[key] = ""
        elif isinstance(val, (str, int, float, bool)):
            meta[key] = val
        else:
            meta[key] = json.dumps(val, ensure_ascii=False)
    return meta



def load_model(name: str = MODEL_NAME, device: str = "cuda") -> SentenceTransformer:
    model = SentenceTransformer(name, device=device)
    model.max_seq_length = MAX_SEQ_LEN
    return model


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = BATCH_SIZE):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   
        show_progress_bar=True,
    )



def build_index(data_path: str, db_dir: str, collection_name: str, model_name: str):
    rows = load_passages(data_path)
    if not rows:
        raise SystemExit(f"No passages found in {data_path}")

    ids = [r["passage_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("Duplicate passage_id values in passages.jsonl")

    raw_texts = [r["text"] for r in rows]
    norm_texts = [norm(t) for t in raw_texts]

    empty = [i for i, t in zip(ids, norm_texts) if not t]
    if empty:
        raise SystemExit(f"Empty text after normalization for: {empty}")

    metadatas = [clean_metadata(r) for r in rows]

    print(f"Loaded {len(rows)} passages. Loading model {model_name} ...")
    model = load_model(model_name)
    embeddings = embed_texts(model, norm_texts)

    client = chromadb.PersistentClient(path=db_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=raw_texts,      
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} passages in '{collection_name}' at {db_dir}/")
    return model


def main():
    parser = argparse.ArgumentParser(description="Build the dense passage index.")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--db-dir", default=DB_DIR)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    build_index(args.data, args.db_dir, args.collection, args.model)


if __name__ == "__main__":
    main()



def dense_search(model, collection, query: str, k: int = 10, where: dict | None = None):
    q_emb = model.encode([norm(query)], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        where=where,
        include=["distances"],
    )
    return [(pid, 1.0 - dist) for pid, dist in zip(res["ids"][0], res["distances"][0])]


