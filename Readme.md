# YASCO Coding Task

This coding task builds a minimum viable retrieval-augmented generation (RAG) system for old Kuwaiti newspapers.

## Repository structure

| Path | Role |
| --- | --- |
| `src/bm25.py` | Builds and searches the BM25 lexical index. |
| `src/common.py` | Defines shared paths and loads JSON Lines files. |
| `src/embed.py` | Builds and searches the Chroma dense-vector index. |
| `src/evals.py` | Runs retrieval evaluation and writes metric tables. |
| `src/generate.py` | Builds the Gradio chat app and generates answers from retrieved context. |
| `src/normalize.py` | Normalizes Arabic text for indexing and search. |
| `src/retrieve.py` | Routes each query type to its retrieval strategy. |
| `requirements.txt` | Lists the Python dependencies. |

## Steps

1. **Prepare the indexes.**

   - Normalize Arabic text by removing diacritics and tatweel and applying character normalization.
   - Build the lexical index using the BM25 algorithm.
   - Encode passages and store their vectors and metadata in Chroma DB.

2. **Retrieve passages.**

   - Make a query router that selects a strategy based on the query type:
     - `fact`: combine BM25 and reranked dense results with reciprocal rank fusion (RRF).
     - `topic`: combine BM25, fuzzy token matches, and reranked dense results with RRF. BM25 is included twice to give lexical matches more weight.
     - `exact_citation`: combine BM25 and fuzzy matches with RRF.
     - `list` and `exhaustive`: combine BM25, fuzzy, and reranked dense results with RRF. Select pages whose score is at least half of the best page score, then rerank the passages from the selected pages.
     - `table_row`: combine BM25 and reranked dense results with RRF. BM25 is included twice to give it more weight. If any table-row passages are found in the fused results, return those table rows instead of the full fused result.
     - `date_range`: extract a date from the query and select pages whose publication date matches it. A month-only date matches every date in that month. Matching pages are then ranked using the maximum hybrid retrieval score found for each page.
     - `abstain`: use the same hybrid retrieval strategy as `fact`, then compare the top fused score with the abstain threshold. If the score is less than or equal to the threshold, mark the query as abstain.

3. **Generate an answer.**

   - Use a tiny model to respond to queries based on the retrieved passages.
   - Build a small chatbot UI using Gradio for demonstration.

## Demonstration

![Gradio demonstration](demo/screenshot.png)

## Evaluation

### Per query metrics

| query | type | tier | k_embed | recall@10 | recall@5 | mrr | abstain_correct@5 | abstain_correct@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | fact | B | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q02 | topic | B | 20 | 0.429 | 0.286 | 1.000 | - | - |
| q03 | fact | B | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q04 | exact_citation | S | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q05 | fact | S | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q06 | list | S | 20 | 0.364 | 0.182 | 1.000 | - | - |
| q07 | table_row | G | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q08 | table_row | G | 20 | 1.000 | 1.000 | 0.333 | - | - |
| q09 | fact | G | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q10 | exhaustive | X | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q11 | date_range | X | 20 | 1.000 | 1.000 | 1.000 | - | - |
| q12 | abstain | X | 20 | - | - | - | 1 | 1 |

### Per tier metrics

| tier | k_embed | n | recall@10 | recall@5 | mrr |
| --- | --- | --- | --- | --- | --- |
| B | 20 | 3 | 0.810 | 0.762 | 1.000 |
| G | 20 | 3 | 1.000 | 1.000 | 0.778 |
| S | 20 | 3 | 0.788 | 0.727 | 1.000 |
| X | 20 | 2 | 1.000 | 1.000 | 1.000 |

### Per query type metrics

| type | k_embed | n | recall@10 | recall@5 | mrr |
| --- | --- | --- | --- | --- | --- |
| date_range | 20 | 1 | 1.000 | 1.000 | 1.000 |
| exact_citation | 20 | 1 | 1.000 | 1.000 | 1.000 |
| exhaustive | 20 | 1 | 1.000 | 1.000 | 1.000 |
| fact | 20 | 4 | 1.000 | 1.000 | 1.000 |
| list | 20 | 1 | 0.364 | 0.182 | 1.000 |
| table_row | 20 | 2 | 1.000 | 1.000 | 0.667 |
| topic | 20 | 1 | 0.429 | 0.286 | 1.000 |

### Special query types

- **Exhaustive:** The system uses a page-selection strategy. It combines BM25, fuzzy, and reranked dense results with RRF. It gets the highest-scoring page result, calculates a threshold that is currently half of that score, then selects pages with scores at least as high as the threshold. Finally, it reranks all passages belonging to the selected pages.

- **Date range:** The system extracts the date from the query using a rule-based approach, selects pages whose publication date matches it, and ranks those pages using hybrid retrieval scores. For a month-only date, all pages whose publication date starts with that month are selected.

- **Abstain:** It uses the same hybrid BM25 + reranked dense strategy as `fact`, then compares the top fused score with the abstain threshold. If the score is less than or equal to the threshold, the query is marked as abstain. The choice of the threshold is heuristic for this demo. For production, we may use another method or optimize it.

## Observed failures

1. **Topic coverage:** Recall@10 is 0.429 and Recall@5 is 0.286. The first relevant passage ranks first, as shown by MRR 1.000, but the result list misses several relevant passages. This is mainly because there are many relevant passages.

2. **List coverage:** Recall@10 is 0.364 and Recall@5 is 0.182. The first relevant passage ranks first, but the result list misses most of the relevant passages. Same problem as topic.

3. **Table-row ordering:** Even though I get good results for Recall@5 and Recall@10, both are 1.000, I didn't do any special work for table layouts. The current strategy only prefers passages marked as table rows when they appear in the fused results. A query asking to summarize a table, for example, will probably fail because the system does not reconstruct the table structure. If two similar tables contain similar information, this may also cause failures.

## Week Plan

If I had one more week, I would:

- Experiment with other strategies for topic and list coverage, such as an expanding-window strategy.
- Handle table layouts by reconstructing tables from consecutive table rows.
- Use a tiny LLM to describe a given table and, if possible, describe a page (e.g., "this page talks about ..."). Then, for topic and list queries, I could search for the specific page first and retrieve its content as context.
- Make date-range extraction more robust to handle Hijri dates and date-range periods.


## Run the full pipeline

From the repository root, run:

```bash
python run.py
```

The script installs `requirements.txt`, rebuilds the BM25 and dense indexes, writes the retrieval evaluation to `results/evaluation.txt`, and then starts the Gradio app. 
