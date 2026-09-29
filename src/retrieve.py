
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from bm25 import BM25Index, tokenize
from normalize import norm
from common import BM25_INDEX_PATH as DEFAULT_BM25_INDEX_PATH
from common import CHROMA_DB_DIR as DEFAULT_CHROMA_DB_DIR, CHROMA_COLLECTION
from rapidfuzz import fuzz
from evals import (
    ABSTAIN_THRESHOLD,
    DEFAULT_K,
    DEFAULT_K_EMBED,
    EXHAUSTIVE_SCORE_RATIO,
    FUZZY_SCORE_THRESHOLD,
    RERANKER_MODEL,
    RRF_K,
)

BM25_INDEX_PATH = str(DEFAULT_BM25_INDEX_PATH)
CHROMA_DB_DIR = str(DEFAULT_CHROMA_DB_DIR)

@dataclass
class RetrievalResult:
    kind: str                      
    items: list = field(default_factory=list)   # [(id, score), ...]
    note: str = ""                 



class Indexes:

    def __init__(self, bm25_path: str = BM25_INDEX_PATH, chroma_dir: str = CHROMA_DB_DIR,
                 chroma_collection: str = CHROMA_COLLECTION, use_dense: bool = True,
                 reranker_name: str = RERANKER_MODEL):
        self.bm25 = BM25Index.load(bm25_path)

        self.meta_by_id = dict(zip(self.bm25.ids, self.bm25.metadatas))
        self.text_by_id = dict(zip(self.bm25.ids, self.bm25.raw_texts))

        self.model = None
        self.reranker = None
        self.collection = None
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(
            reranker_name,
            device="cuda",
            trust_remote_code=True,
        )
        if use_dense:
            import chromadb
            from embed import load_model
            self.model = load_model()
            self.collection = chromadb.PersistentClient(path=chroma_dir).get_collection(chroma_collection)


    def bm25_search(self, query: str, k: int = 50, where: dict | None = None):
        return self.bm25.search(query, k=k, where=where)

    def dense_search(self, query: str, k: int = 50, where: dict | None = None):
        if self.collection is None:
            return []
        from embed import dense_search
        return dense_search(self.model, self.collection, query, k=k, where=where)

    def reranked_dense_search(self, query: str, k_embed: int, k_rerank: int) -> list[tuple[str, float]]:
        candidates = self.dense_search(query, k=k_embed)
        return self.rerank_passages(query, [pid for pid, _ in candidates], k_rerank)

    def rerank_passages(self, query: str, passage_ids: list[str], k: int) -> list[tuple[str, float]]:
        if not passage_ids:
            return []
        passages = [self.text_by_id[pid] for pid in passage_ids]
        pairs = [(query, text) for text in passages]
        scores = self.reranker.predict(pairs)
        ranked = [(passage_ids[i], float(score)) for i, score in enumerate(scores)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]

def rrf(rankings: list[list[tuple[str, float]]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion"""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (pid, _score) in enumerate(ranking):
            fused[pid] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# --- query understanding ---------------------------------------------------

_ARABIC_MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "أبريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "اغسطس": 8, "أغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "أكتوبر": 10,
    "نوفمبر": 11, "ديسمبر": 12,
}
_DATE_RE = re.compile(
    r"(\d{1,2})?\s*(" + "|".join(_ARABIC_MONTHS) + r")\s*(\d{4})"
)
_NUMERIC_DATE_RE = re.compile(
    r"(?P<year>\d{4})[-/.](?P<month>\d{1,2})(?:[-/.](?P<day>\d{1,2}))?"
    r"|(?P<day_first>\d{1,2})[-/.](?P<month_first>\d{1,2})[-/.](?P<year_last>\d{4})"
)
def extract_date(query: str):
    """Extract a date as YYYY-MM-DD, YYYY-MM, or return None."""
    text = norm(query)
    match = _NUMERIC_DATE_RE.search(text)
    if match:
        parts = match.groupdict()
        year = parts["year"] or parts["year_last"]
        month = parts["month"] or parts["month_first"]
        day = parts["day"] or parts["day_first"]
        return _format_date(year, month, day)

    match = _DATE_RE.search(text)
    if match:
        day, month_name, year = match.groups()
        return _format_date(year, _ARABIC_MONTHS[month_name], day)
    return None


def _format_date(year: str, month: str | int, day: str | None = None) -> str | None:
    year_num, month_num = int(year), int(month)
    if not 1 <= month_num <= 12:
        return None
    if day is None:
        return f"{year_num:04d}-{month_num:02d}"
    day_num = int(day)
    try:
        date(year_num, month_num, day_num)
    except ValueError:
        return None
    return f"{year_num:04d}-{month_num:02d}-{day_num:02d}"


def fuzzy_search(idx: Indexes, query: str, k: int = 50) -> list[tuple[str, float]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    matches = []
    for passage_id in idx.bm25.ids:
        passage_tokens = tokenize(idx.text_by_id[passage_id])
        score = max((fuzz.partial_ratio(q, p) for q in query_tokens for p in passage_tokens), default=0)
        if score >= FUZZY_SCORE_THRESHOLD:
            matches.append((passage_id, float(score)))
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[:k]


# --- per-type strategies ----------------------------------------------------

def strategy_fact(idx: Indexes, query: str, k_embed: int, k_rerank: int) -> RetrievalResult:
    """hybrid: BM25 + dense"""
    bm = idx.bm25_search(query, k=k_rerank)
    dense = idx.reranked_dense_search(query, k_embed, k_rerank)
    fused = rrf([bm, dense]) if dense else bm
    return RetrievalResult("passages", fused[:k_rerank], note="BM25 + reranked dense RRF")


def strategy_exact_citation(idx: Indexes, query: str, k_embed: int, k_rerank: int) -> RetrievalResult:
    """Combine lexical and fuzzy matching for citation-like queries."""
    bm = idx.bm25_search(query, k=k_rerank)
    fuzzy = fuzzy_search(idx, query, k=k_rerank)
    fused = rrf([bm, fuzzy]) if fuzzy else bm
    return RetrievalResult("passages", fused[:k_rerank], note="lexical + fuzzy RRF")


def strategy_topic(idx: Indexes, query: str, k_embed: int, k_rerank: int) -> RetrievalResult:
    """Fuse lexical, fuzzy-token, and dense rankings for short topics."""
    bm = idx.bm25_search(query, k=k_rerank)
    fuzzy = fuzzy_search(idx, query, k=k_rerank)
    dense = idx.reranked_dense_search(query, k_embed, k_rerank)
    fused = rrf([ranking for ranking in (bm,bm, fuzzy, dense) if ranking])
    return RetrievalResult("passages", fused[:k_rerank], note="lexical + fuzzy + reranked dense RRF")


def strategy_page_rerank(idx: Indexes, query: str, k_embed: int, k_rerank: int) -> RetrievalResult:
    """Select relevant pages from hybrid hits, then rerank their passages."""
    bm = idx.bm25_search(query, k=k_rerank)
    fuzzy = fuzzy_search(idx, query, k=k_rerank)
    dense = idx.reranked_dense_search(query, k_embed, k_rerank)
    fused = rrf([ranking for ranking in (bm, fuzzy, dense) if ranking])
    if not fused:
        return RetrievalResult("passages", [], note="no passage matches")

    page_scores: dict[str, float] = defaultdict(float)
    for passage_id, score in fused:
        page_id = idx.meta_by_id[passage_id]["page_id"]
        page_scores[page_id] = max(page_scores[page_id], score)

    threshold = EXHAUSTIVE_SCORE_RATIO * max(page_scores.values())
    selected_pages = {page_id for page_id, score in page_scores.items() if score >= threshold}
    page_passages = [
        passage_id for passage_id in idx.bm25.ids
        if idx.meta_by_id[passage_id]["page_id"] in selected_pages
    ]
    reranked = idx.rerank_passages(query, page_passages, k_rerank)
    return RetrievalResult(
        "passages", reranked,
        note=f"hybrid page threshold={threshold:.4f}; reranked {len(page_passages)} passages",
    )


def strategy_table_row(idx: Indexes, query: str, k_embed: int, k_rerank: int) -> RetrievalResult:
    """Favor lexical matches when retrieving table rows."""
    bm = idx.bm25_search(query, k=k_rerank)
    dense = idx.reranked_dense_search(query, k_embed, k_rerank)
    fused = rrf([bm, bm, dense]) if dense else bm

    table_only = [(pid, s) for pid, s in fused if idx.meta_by_id[pid].get("is_table_row")]
    if table_only:
        return RetrievalResult("passages", table_only[:k_rerank], note="BM25-weighted hybrid, table rows")
    return RetrievalResult("passages", fused[:k_rerank], note="BM25-weighted hybrid")


def strategy_date_range(idx: Indexes, query: str, k_embed: int, k_rerank: int) -> RetrievalResult:
    """Filter pages by date, then rank matching pages with hybrid retrieval."""
    date = extract_date(query)
    if not date:
        return RetrievalResult("pages", [], note="no date found in query")

    month_only = len(date) == 7

    def matches(publication_date):
        if not isinstance(publication_date, str):
            return False
        return publication_date.startswith(date) if month_only else publication_date == date

    candidate_pages = {
        meta["page_id"] for meta in idx.meta_by_id.values() if matches(meta["publication_date"])
    }
    if not candidate_pages:
        return RetrievalResult("pages", [], note=f"no pages with publication_date matching {date}")

    bm = idx.bm25_search(query, k=k_rerank)
    dense = idx.reranked_dense_search(query, k_embed, k_rerank)
    fused = rrf([bm, dense]) if dense else bm

    page_score: dict[str, float] = defaultdict(float)
    for pid, score in fused:
        page_id = idx.meta_by_id[pid]["page_id"]
        if page_id in candidate_pages:
            page_score[page_id] = max(page_score[page_id], score)
    # Pages with no lexical/semantic hit still count as "published that day".
    for page_id in candidate_pages:
        page_score.setdefault(page_id, 0.0)

    ranked = sorted(page_score.items(), key=lambda x: x[1], reverse=True)
    return RetrievalResult("pages", ranked, note=f"date filter={date}, {len(candidate_pages)} candidate pages")


STRATEGIES = {
    "fact": strategy_fact,
    "topic": strategy_topic,
    "exact_citation": strategy_exact_citation,
    "list": strategy_page_rerank,
    "table_row": strategy_table_row,
    "exhaustive": strategy_page_rerank,
    "date_range": strategy_date_range,
    "abstain": strategy_fact,
}



def retrieve(idx: Indexes, query: str, qtype: str, k_embed: int = DEFAULT_K_EMBED,
             k_rerank: int = DEFAULT_K) -> tuple[str, RetrievalResult]:
    """Run the strategy selected by the required query type."""
    if not qtype:
        raise ValueError("qtype is required")
    if k_embed < 1 or k_rerank < 1 or k_rerank > k_embed:
        raise ValueError("k_embed and k_rerank must be positive, with k_rerank <= k_embed")
    try:
        strategy = STRATEGIES[qtype]
    except KeyError as exc:
        raise ValueError(f"Unknown query type {qtype!r}; expected one of {', '.join(STRATEGIES)}") from exc
    return qtype, strategy(idx, query, k_embed, k_rerank)
