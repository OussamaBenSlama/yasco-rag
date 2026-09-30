import argparse
from itertools import product
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from common import (
    BM25_INDEX_PATH,
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
    QUERIES_PATH,
    ROOT,
    load_jsonl,
)

if TYPE_CHECKING:
    from retrieve import Indexes

RRF_K = 60
DEFAULT_K = 10
ABSTAIN_THRESHOLD = 0.035
EXHAUSTIVE_SCORE_RATIO = 0.5
FUZZY_SCORE_THRESHOLD = 20
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_K_EMBED = 20
K_EMBED_VALUES = (20,)
K_RERANK_VALUES = (5, 10)


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant:
            return 1 / rank
    return 0.0


def evaluate_query(
    query: dict,
    indexes: "Indexes",
    k_embed: int,
    k_rerank: int,
    report_k_embed: int | None = None,
) -> dict:
    from retrieve import retrieve

    query_type = query["type"]
    _, result = retrieve(indexes, query["query"], qtype=query_type,
                         k_embed=k_embed, k_rerank=k_rerank)
    expected_abstain = query.get("expected") == "abstain"
    abstain_correct = None

    if expected_abstain:
        abstain_correct = result.kind == "abstain"
        recall5 = recall10 = mrr = None
    else:
        relevant_pages = set(query.get("relevant_pages") or [])
        if result.kind == "pages":
            relevant = relevant_pages
            ranked_ids = [page_id for page_id, _ in result.items]
        elif relevant_pages:
            relevant = relevant_pages
            ranked_ids = list(dict.fromkeys(
                indexes.meta_by_id[passage_id]["page_id"]
                for passage_id, _ in result.items
                if passage_id in indexes.meta_by_id
            ))
        else:
            relevant = set(query.get("relevant_passages") or [])
            ranked_ids = [passage_id for passage_id, _ in result.items]

        if result.kind == "abstain":
            recall5 = 0.0 if relevant and k_rerank >= 5 else None
            recall10 = 0.0 if relevant and k_rerank >= 10 else None
            mrr = 0.0 if relevant else None
        else:
            recall5 = recall_at_k(ranked_ids, relevant, 5) if k_rerank >= 5 else None
            recall10 = recall_at_k(ranked_ids, relevant, 10) if k_rerank >= 10 else None
            mrr = reciprocal_rank(ranked_ids, relevant)

    return {
        "query": query.get("query_id", "?"),
        "type": query_type,
        "tier": query.get("tier", "?"),
        "k_embed": report_k_embed,
        "k_rerank": k_rerank,
        "recall@10": recall10,
        "recall@5": recall5,
        "mrr": mrr,
        "expected_abstain": expected_abstain,
        "abstain_correct": int(abstain_correct) if abstain_correct is not None else None,
    }


def aggregate(rows: list[dict], group: str) -> list[dict]:
    values: dict[tuple, list[dict]] = {}
    for row in rows:
        if not row["expected_abstain"]:
            key = (row[group], row["k_embed"])
            values.setdefault(key, []).append(row)

    summaries = []
    for (value, k_embed), group_rows in sorted(
        values.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        summary = {
            group: value,
            "k_embed": k_embed,
            "n": len({row["query"] for row in group_rows}),
            "recall@10": average_metric(group_rows, "recall@10", 10),
            "recall@5": average_metric(group_rows, "recall@5", 5),
            "mrr": average_metric(group_rows, "mrr", 10),
        }
        summaries.append(summary)
    return summaries


def average_metric(rows: list[dict], metric: str, k_rerank: int) -> float | None:
    scores = [
        row[metric] for row in rows
        if row["k_rerank"] == k_rerank and row[metric] is not None
    ]
    return round(mean(scores), 3) if scores else None


def combine_pair_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    keys = ("query", "type", "tier", "k_embed")
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    combined = []
    for key, query_rows in grouped.items():
        combined.append({
            **dict(zip(keys, key)),
            "recall@10": next((row["recall@10"] for row in query_rows if row["k_rerank"] == 10), None),
            "recall@5": next((row["recall@5"] for row in query_rows if row["k_rerank"] == 5), None),
            "mrr": next((row["mrr"] for row in query_rows if row["k_rerank"] == 10), None),
            "abstain_correct": next(
                (row.get("abstain_correct") for row in query_rows
                 if row.get("abstain_correct") is not None),
                None,
            ),
            "abstain_correct@5": next(
                (row.get("abstain_correct") for row in query_rows
                 if row["k_rerank"] == 5),
                None,
            ),
            "abstain_correct@10": next(
                (row.get("abstain_correct") for row in query_rows
                 if row["k_rerank"] == 10),
                None,
            ),
        })
    return combined


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_table(headers: list[str], rows: list[dict]) -> str:
    values = [[format_value(row.get(header)) for header in headers] for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in values)]) for i, header in enumerate(headers)]
    lines = [" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in values)
    return "\n".join(lines)


def build_report(rows: list[dict]) -> str:
    combined_rows = combine_pair_rows(rows)
    query_rows = [
        {key: row.get(key) for key in (
            "query", "type", "tier", "k_embed", "recall@10", "recall@5", "mrr",
            "abstain_correct@5", "abstain_correct@10",
        )}
        for row in combined_rows
    ]
    tier_headers = ["tier", "k_embed", "n", "recall@10", "recall@5", "mrr"]
    type_headers = ["type", "k_embed", "n", "recall@10", "recall@5", "mrr"]
    sections = [
        ("Per query metrics", format_table(
            ["query", "type", "tier", "k_embed", "recall@10", "recall@5", "mrr",
             "abstain_correct@5", "abstain_correct@10"], query_rows
        )),
        ("Per tier metrics", format_table(tier_headers, aggregate(rows, "tier"))),
        ("Per query type metrics", format_table(type_headers, aggregate(rows, "type"))),
        ("Abstention metrics", format_table(
            ["k_embed", "k_rerank", "queries", "correct", "accuracy"],
            _abstention_summary(rows),
        )),
    ]
    return "\n\n".join(f"{title}\n{table}" for title, table in sections) + "\n"


def _abstention_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row["expected_abstain"] and row.get("abstain_correct") is not None:
            key = (row["k_embed"], row["k_rerank"])
            groups.setdefault(key, []).append(row)

    summaries = []
    for (k_embed, k_rerank), group_rows in sorted(groups.items()):
        # Count each query once for each retrieval configuration.
        outcomes = {row["query"]: row["abstain_correct"] for row in group_rows}
        correct = sum(outcomes.values())
        total = len(outcomes)
        summaries.append({
            "k_embed": k_embed,
            "k_rerank": k_rerank,
            "queries": total,
            "correct": correct,
            "accuracy": round(correct / total, 3) if total else None,
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and write a text report.")
    parser.add_argument("--queries", default=str(QUERIES_PATH))
    parser.add_argument("--k-embed", type=int, nargs="+", default=K_EMBED_VALUES)
    parser.add_argument("--k-rerank", type=int, nargs="+", choices=K_RERANK_VALUES,
                        default=K_RERANK_VALUES)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "results" / "evaluation.txt"))
    parser.add_argument("--bm25-index", default=str(BM25_INDEX_PATH))
    parser.add_argument("--chroma-db", default=str(CHROMA_DB_DIR))
    parser.add_argument("--chroma-collection", default=CHROMA_COLLECTION)
    args = parser.parse_args()

    from retrieve import Indexes

    indexes = Indexes(
        args.bm25_index,
        args.chroma_db,
        args.chroma_collection,
        use_dense=not args.no_dense,
    )
    queries = load_jsonl(args.queries)
    pairs = [(k_embed, k_rerank, k_embed) for k_embed, k_rerank in product(
        args.k_embed, args.k_rerank
    )] if not args.no_dense else [
        (DEFAULT_K_EMBED, k_rerank, None) for k_rerank in args.k_rerank
    ]
    rows = [
        evaluate_query(query, indexes, k_embed, k_rerank, report_k_embed)
        for k_embed, k_rerank, report_k_embed in pairs
        for query in queries
    ]
    report = build_report(rows)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
