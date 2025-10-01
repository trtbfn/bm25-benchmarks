"""Run BM25S on the MS MARCO dev split with multi-threaded batch retrieval.

The script downloads the dataset if necessary, builds the BM25S index, and
retrieves results in batches while using the requested number of threads.  It
records standard BEIR metrics as well as timing/throughput statistics for
later analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable, Sequence, Tuple

import numpy as np
import Stemmer

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import bm25s
from bm25s.utils.benchmark import Timer, get_max_memory_usage
from bm25s.utils.beir import (
    BASE_URL,
    clean_results_keys,
    merge_cqa_dupstack,
    postprocess_results_for_eval,
)


_STOPWORD_PRESETS = {
    "en": tuple(sorted(bm25s.tokenization.STOPWORDS_EN)),
    "english": tuple(sorted(bm25s.tokenization.STOPWORDS_EN)),
}


def resolve_stopwords(option: str | None) -> tuple[Sequence[str] | None, str]:
    """Map a command-line stopword option to a concrete list of tokens."""

    if option is None or option.lower() == "none":
        return None, "none"

    key = option.lower()
    if key in _STOPWORD_PRESETS:
        canonical = "en" if key != "en" else key
        return list(_STOPWORD_PRESETS[key]), canonical

    raise ValueError(f"Unsupported stopword option: {option}")


def batched_indices(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) index pairs covering ``range(n_items)``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def retrieve_in_batches(
    model: bm25s.BM25,
    queries: Sequence[Sequence[int]],
    corpus_ids: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
    n_threads: int,
    backend_selection: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve ``top_k`` documents for the provided queries in batches."""
    results_batches = []
    scores_batches = []

    for start, end in batched_indices(len(queries), batch_size):
        query_batch = queries[start:end]
        queried_results, queried_scores = model.retrieve(
            query_batch,
            corpus=corpus_ids,
            k=top_k,
            return_as="tuple",
            n_threads=n_threads,
            backend_selection=backend_selection,
        )
        results_batches.append(queried_results)
        scores_batches.append(queried_scores)

    return np.vstack(results_batches), np.vstack(scores_batches)


def main(
    *,
    save_dir: Path,
    result_dir: Path,
    n_threads: int,
    batch_size: int,
    top_k: int,
    method: str,
    idf_method: str,
    k1: float,
    b: float,
    stopwords: str | None,
    stemmer_name: str | None,
) -> Path:
    dataset = "msmarco"

    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), str(save_dir))

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    split = "dev"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    corpus_ids = np.array(list(corpus.keys()))
    corpus_texts = [f"{item['title']} {item['text']}" for item in corpus.values()]
    num_docs = len(corpus_ids)

    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    stopword_list, stopwords_label = resolve_stopwords(stopwords)
    stemmer_name = None if stemmer_name == "none" else stemmer_name
    stemmer = Stemmer.Stemmer("english") if stemmer_name == "snowball" else None

    timer = Timer("[bm25s-msmarco]")

    tokenizer = bm25s.tokenization.Tokenizer(stopwords=stopword_list, stemmer=stemmer)

    t = timer.start("Tokenize corpus")
    corpus_tokenized = tokenizer.tokenize(corpus_texts, update_vocab=True, return_as="tuple")
    timer.stop(t, show=True, n_total=num_docs)

    t = timer.start("Tokenize queries")
    queries_tokenized = tokenizer.tokenize(query_texts, update_vocab=False, return_as="ids")
    timer.stop(t, show=True, n_total=len(query_texts))

    model = bm25s.BM25(method=method, idf_method=idf_method, k1=k1, b=b)

    t = timer.start("Index")
    model.index(corpus_tokenized, leave_progress=False)
    timer.stop(t, show=True, n_total=num_docs)

    model.activate_numba_scorer()
    model.get_scores(queries_tokenized[0])

    timer.start("Batch retrieve")
    start_time = time.perf_counter()
    queried_results, queried_scores = retrieve_in_batches(
        model,
        queries_tokenized,
        corpus_ids,
        batch_size=batch_size,
        top_k=top_k,
        n_threads=n_threads,
        backend_selection="numba",
    )
    retrieval_time = time.perf_counter() - start_time
    timer.stop("Batch retrieve", show=True, n_total=len(query_texts))

    results_dict = postprocess_results_for_eval(queried_results, queried_scores, query_ids)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_dict, [1, 10, 100, 1000])

    max_mem_gb = get_max_memory_usage("GB")
    qps = len(query_texts) / retrieval_time if retrieval_time > 0 else 0.0

    print("=" * 50)
    print("Dataset:", dataset)
    print(f"Queries: {len(query_texts):,}")
    print(f"Docs: {num_docs:,}")
    print(f"Threads: {n_threads}")
    print(f"Batch size: {batch_size}")
    print(f"Retrieval time: {retrieval_time:.2f} s ({qps:.2f} q/s)")
    print(f"Max memory usage: {max_mem_gb:.2f} GB")
    print("-" * 50)
    print(ndcg)
    print(recall)
    print("=" * 50)

    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = result_dir / f"bm25s-msmarco-{time.strftime('%Y%m%d-%H%M%S')}.json"

    save_dict = {
        "model": "bm25s",
        "dataset": dataset,
        "method": method,
        "idf_method": idf_method,
        "k1": k1,
        "b": b,
        "stopwords": stopwords_label,
        "stemmer": stemmer_name or "none",
        "n_threads": n_threads,
        "batch_size": batch_size,
        "top_k": top_k,
        "retrieval_time": retrieval_time,
        "queries_per_second": qps,
        "max_mem_gb": max_mem_gb,
        "timing": timer.to_dict(underscore=True, lowercase=True),
        "stats": {
            "num_docs": num_docs,
            "num_queries": len(query_texts),
        },
        "scores": {
            "ndcg": clean_results_keys(ndcg),
            "map": clean_results_keys(_map),
            "recall": clean_results_keys(recall),
            "precision": clean_results_keys(precision),
        },
    }

    with save_path.open("w") as f:
        json.dump(save_dict, f, indent=2)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-threaded BM25S retrieval on MS MARCO in batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save_dir", type=Path, default=Path("datasets"))
    parser.add_argument("--result_dir", type=Path, default=Path("results/bm25s"))
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument(
        "--method",
        type=str,
        default="lucene",
        choices=["lucene", "atire", "robertson", "bm25l", "bm25+"],
    )
    parser.add_argument(
        "--idf_method",
        type=str,
        default="lucene",
        choices=["lucene", "bm25", "atire", "robertson"],
    )
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--stopwords", type=str, default="en", choices=["en", "english", "none"])
    parser.add_argument(
        "--stemmer_name",
        type=str,
        default="snowball",
        choices=["snowball", "none"],
    )

    args = parser.parse_args()

    save_path = main(
        save_dir=args.save_dir,
        result_dir=args.result_dir,
        n_threads=args.n_threads,
        batch_size=args.batch_size,
        top_k=args.top_k,
        method=args.method,
        idf_method=args.idf_method,
        k1=args.k1,
        b=args.b,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer_name,
    )

    print(f"Saved results to {save_path}")
