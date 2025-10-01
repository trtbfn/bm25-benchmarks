"""Run SketchBM25 on the MS MARCO dev split with multi-threaded batch retrieval.

The script mirrors the BM25S benchmarking entry point by downloading the
dataset, building the index, and retrieving results in batches while using the
requested number of worker threads. Standard BEIR metrics together with timing,
latency, and throughput statistics are persisted to disk for later analysis.

Compared to the initial version this variant adds configurable IDF formulations,
optional negative IDF support, richer English stopword lists, and an optional
Numba-accelerated top-k selection to better match the SciFact all-in-one
runner.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import Stemmer

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from bm25s import tokenization as bm25s_tokenization
from bm25s.utils.benchmark import Timer, get_max_memory_usage
from bm25s.utils.beir import BASE_URL, merge_cqa_dupstack

try:
    from stopwords import STOPWORDS_EN_PLUS as _RICH_EN_STOPWORDS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _RICH_EN_STOPWORDS = None

try:
    from numba import njit

    _NUMBA_OK = True
except Exception:  # pragma: no cover - optional dependency
    _NUMBA_OK = False


def batched_indices(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, end)`` index pairs covering ``range(n_items)``."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


_TOKEN_PATTERN = re.compile(r"\W+")
_DEFAULT_STOPWORDS = (
    tuple(sorted(_RICH_EN_STOPWORDS)) if _RICH_EN_STOPWORDS else tuple(sorted(bm25s_tokenization.STOPWORDS_EN))
)
_STOPWORD_PRESETS = {
    "en": _DEFAULT_STOPWORDS,
    "english": _DEFAULT_STOPWORDS,
}
if _RICH_EN_STOPWORDS:
    _STOPWORD_PRESETS["en_plus"] = tuple(sorted(_RICH_EN_STOPWORDS))
    _STOPWORD_PRESETS["english_plus"] = tuple(sorted(_RICH_EN_STOPWORDS))


if _NUMBA_OK:

    @njit()
    def _sift_down(values, indices, startpos, pos):
        new_value = values[pos]
        new_index = indices[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent_value = values[parentpos]
            if new_value < parent_value:
                values[pos] = parent_value
                indices[pos] = indices[parentpos]
                pos = parentpos
                continue
            break
        values[pos] = new_value
        indices[pos] = new_index


    @njit()
    def _sift_up(values, indices, pos, length):
        startpos = pos
        new_value = values[pos]
        new_index = indices[pos]
        childpos = 2 * pos + 1
        while childpos < length:
            rightpos = childpos + 1
            if rightpos < length and values[rightpos] < values[childpos]:
                childpos = rightpos
            values[pos] = values[childpos]
            indices[pos] = indices[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        values[pos] = new_value
        indices[pos] = new_index
        _sift_down(values, indices, startpos, pos)


    @njit()
    def _numba_sorted_top_k(array: np.ndarray, k: int, sorted=True):
        n = len(array)
        if k > n:
            k = n
        values = np.zeros(k, dtype=array.dtype)
        indices = np.zeros(k, dtype=np.int32)
        length = 0
        for i, value in enumerate(array):
            if length < k:
                values[length] = value
                indices[length] = i
                _sift_down(values, indices, 0, length)
                length += 1
            else:
                if value > values[0]:
                    values[0] = value
                    indices[0] = i
                    _sift_up(values, indices, 0, length)
        if sorted:
            order = np.argsort(values)[::-1]
            values = values[order]
            indices = indices[order]
        return values, indices


def resolve_stopwords(option: str | None) -> tuple[Optional[set[str]], str]:
    """Map a stopword CLI option to a concrete filter set."""

    if option is None or option.lower() == "none":
        return None, "none"

    key = option.lower()
    if key in _STOPWORD_PRESETS:
        if key == "english":
            canonical = "en"
        elif key == "english_plus":
            canonical = "en_plus"
        else:
            canonical = key
        return set(_STOPWORD_PRESETS[key]), canonical

    raise ValueError(
        f"Unsupported stopword option: {option}. Available: {', '.join(sorted(_STOPWORD_PRESETS))}, none"
    )


class Tokenizer:
    """Simple tokeniser with optional stopword filtering and stemming."""

    def __init__(
        self,
        *,
        lowercase: bool = True,
        stopwords: Optional[Iterable[str]] = None,
        min_len: int = 2,
        stemmer_name: Optional[str] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
    ) -> None:
        self.lowercase = lowercase
        self.stopwords = set(stopwords) if stopwords else None
        self.min_len = min_len
        self.stemmer_name = stemmer_name
        if stemmer is not None:
            self.stemmer = stemmer
            if self.stemmer_name is None:
                self.stemmer_name = "custom"
        elif stemmer_name == "snowball":
            self.stemmer = Stemmer.Stemmer("english")
        else:
            self.stemmer = None

    def __call__(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        tokens = [token for token in _TOKEN_PATTERN.split(text) if token]
        if self.min_len > 1:
            tokens = [token for token in tokens if len(token) >= self.min_len]
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        if self.stemmer is not None:
            tokens = [self.stemmer.stemWord(token) for token in tokens]
        return tokens


class CMS:
    """Count-Min sketch used for approximate document frequencies."""

    def __init__(self, rows: int, cols: int, seeds: List[int]) -> None:
        if len(seeds) < rows or len(set(seeds)) != len(seeds):
            raise ValueError("Invalid CMS seeds")
        self.rows = rows
        self.cols = cols
        self.seeds = seeds[:rows]
        self.matrix = np.zeros((rows, cols), dtype=np.uint32)

    def update(self, token: str, count: int = 1) -> None:
        for row in range(self.rows):
            column = (hash((self.seeds[row], token)) & 0xFFFFFFFF) % self.cols
            self.matrix[row, column] += count

    def query(self, token: str) -> float:
        estimates = []
        for row in range(self.rows):
            column = (hash((self.seeds[row], token)) & 0xFFFFFFFF) % self.cols
            estimates.append(self.matrix[row, column])
        return float(min(estimates) if estimates else 0)

    @staticmethod
    def compute_width_depth(epsilon: float, delta: float) -> Tuple[int, int]:
        width = math.ceil(math.e / max(1e-12, epsilon))
        depth = math.ceil(math.log(1.0 / max(1e-12, delta)))
        return width, depth


class CS:
    """Count sketch variant for signed document-frequency estimation."""

    def __init__(self, rows: int, cols: int, idx_seeds: List[int], sign_seeds: List[int]) -> None:
        if len(idx_seeds) < rows or len(sign_seeds) < rows:
            raise ValueError("Invalid CS seeds")
        if len(set(idx_seeds)) != len(idx_seeds) or len(set(sign_seeds)) != len(sign_seeds):
            raise ValueError("Seeds must be distinct")
        self.rows = rows
        self.cols = cols
        self.idx_seeds = idx_seeds[:rows]
        self.sign_seeds = sign_seeds[:rows]
        self.matrix = np.zeros((rows, cols), dtype=np.int64)

    def _sign(self, token: str, row: int) -> int:
        return 1 if ((hash((self.sign_seeds[row], token)) & 1) == 0) else -1

    def update(self, token: str, count: int = 1) -> None:
        for row in range(self.rows):
            column = (hash((self.idx_seeds[row], token)) & 0xFFFFFFFF) % self.cols
            self.matrix[row, column] += self._sign(token, row) * count

    def query(self, token: str) -> float:
        estimates = []
        for row in range(self.rows):
            column = (hash((self.idx_seeds[row], token)) & 0xFFFFFFFF) % self.cols
            estimates.append(self._sign(token, row) * self.matrix[row, column])
        return float(np.median(estimates)) if estimates else 0.0

    @staticmethod
    def compute_width_depth(epsilon: float, delta: float) -> Tuple[int, int]:
        width = math.ceil(1.0 / max(1e-12, epsilon) ** 2)
        depth = math.ceil(math.log(1.0 / max(1e-12, delta)))
        return width, depth


_TF_CTX: Dict[str, object] = {}


def _tf_worker_init(conf: Dict[str, object]) -> None:
    global _TF_CTX
    _TF_CTX = conf


def _tf_worker_run(args: Tuple[List[int], List[str]]):
    idxs, texts = args
    tokenizer = Tokenizer(**_TF_CTX["tokenizer_conf"])  # type: ignore[arg-type]
    outputs: List[Tuple[int, int, Dict[str, int]]] = []
    for index in idxs:
        tokens = tokenizer(texts[index])
        doc_len = len(tokens)
        term_freqs: Dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        outputs.append((index, doc_len, term_freqs))
    return outputs


def _distinct_seeds(n: int) -> List[int]:
    rng = np.random.default_rng(42)
    seen: set[int] = set()
    seeds: List[int] = []
    while len(seeds) < n:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)
    return seeds


def _chunks(indices: List[int], chunk_size: int) -> List[List[int]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


class SketchBM25:
    """Light-weight BM25 variant with optional sketch-based IDF estimates."""

    def __init__(
        self,
        *,
        k: float = 1.2,
        b: float = 0.75,
        workers: int = 1,
        tokenizer: Optional[Tokenizer] = None,
        sidf_mode: str = "none",
        epsilon: float = 0.01,
        delta: float = 0.01,
        clamp_idf_nonneg: bool = True,
        idf_method: str = "robertson",
        allow_negative_idf: bool = False,
    ) -> None:
        self.k = float(k)
        self.b = float(b)
        self.workers = max(1, int(workers))
        self.tokenizer = tokenizer or Tokenizer()
        self.sidf_mode = sidf_mode
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.clamp_idf_nonneg = bool(clamp_idf_nonneg)
        if idf_method not in {"robertson", "lucene", "atire"}:
            raise ValueError(f"Unsupported idf_method: {idf_method}")
        self.idf_method = idf_method
        self.allow_negative_idf = bool(allow_negative_idf)

        self.doc_names: List[str] = []
        self.doc_len: Optional[np.ndarray] = None
        self.avg_dl: float = 1.0
        self.term_vocab: List[str] = []
        self.term_to_id: Dict[str, int] = {}
        self.term_ptrs: Optional[np.ndarray] = None
        self.post_docids: Optional[np.ndarray] = None
        self.post_tfs: Optional[np.ndarray] = None
        self.cms: Optional[CMS] = None
        self.cs: Optional[CS] = None

    def fit_iter(self, pairs: Iterable[Tuple[str, str]]) -> None:
        names: List[str] = []
        texts: List[str] = []
        for name, text in pairs:
            names.append(str(name))
            texts.append(str(text))

        num_docs = len(names)
        if num_docs == 0:
            raise ValueError("No documents provided to SketchBM25")

        self.doc_names = names

        indices = list(range(num_docs))
        chunk_size = max(1, num_docs // max(1, self.workers))
        chunks = _chunks(indices, chunk_size)
        init_conf = {
            "tokenizer_conf": {
                "lowercase": self.tokenizer.lowercase,
                "min_len": self.tokenizer.min_len,
                "stopwords": list(self.tokenizer.stopwords) if self.tokenizer.stopwords else None,
                "stemmer_name": self.tokenizer.stemmer_name,
            }
        }

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        if self.workers > 1:
            with mp.Pool(processes=self.workers, initializer=_tf_worker_init, initargs=(init_conf,)) as pool:
                tf_batches = pool.map(_tf_worker_run, [(chunk, texts) for chunk in chunks])
        else:
            _tf_worker_init(init_conf)
            tf_batches = [_tf_worker_run((chunk, texts)) for chunk in chunks]

        term_to_id: Dict[str, int] = {}
        term_vocab: List[str] = []
        postings_docs: List[List[int]] = []
        postings_tfs: List[List[float]] = []
        doc_len = np.zeros(num_docs, dtype=np.int32)

        if self.sidf_mode != "none":
            if self.sidf_mode == "cms":
                width, depth = CMS.compute_width_depth(self.epsilon, self.delta)
                self.cms = CMS(rows=depth, cols=width, seeds=_distinct_seeds(depth))
            elif self.sidf_mode == "cs":
                width, depth = CS.compute_width_depth(self.epsilon, self.delta)
                self.cs = CS(
                    rows=depth,
                    cols=width,
                    idx_seeds=_distinct_seeds(depth),
                    sign_seeds=_distinct_seeds(depth),
                )

        for batch in tf_batches:
            for doc_index, doc_length, tf_map in batch:
                doc_len[doc_index] = int(doc_length)
                for term, tf in tf_map.items():
                    term_id = term_to_id.get(term)
                    if term_id is None:
                        term_id = len(term_vocab)
                        term_to_id[term] = term_id
                        term_vocab.append(term)
                        postings_docs.append([])
                        postings_tfs.append([])
                    postings_docs[term_id].append(doc_index)
                    postings_tfs[term_id].append(float(tf))
                    if self.cms is not None:
                        self.cms.update(term, 1)
                    if self.cs is not None:
                        self.cs.update(term, 1)

        self.doc_len = doc_len
        self.avg_dl = float(doc_len.mean()) if num_docs else 1.0
        self.term_vocab = term_vocab
        self.term_to_id = term_to_id

        vocab_size = len(term_vocab)
        ptrs = np.zeros(vocab_size + 1, dtype=np.int64)
        total = 0
        for i in range(vocab_size):
            total += len(postings_docs[i])
            ptrs[i + 1] = total

        docids = np.zeros(total, dtype=np.int32)
        tfs = np.zeros(total, dtype=np.float32)
        for i in range(vocab_size):
            start = ptrs[i]
            end = ptrs[i + 1]
            if start == end:
                continue
            docids[start:end] = np.asarray(postings_docs[i], dtype=np.int32)
            tfs[start:end] = np.asarray(postings_tfs[i], dtype=np.float32)

        self.term_ptrs = ptrs
        self.post_docids = docids
        self.post_tfs = tfs

    def _idf(self, df_raw: float, num_docs: int, term: str) -> float:
        if self.sidf_mode == "cms" and self.cms is not None:
            df = max(1.0, self.cms.query(term))
        elif self.sidf_mode == "cs" and self.cs is not None:
            df = max(1.0, self.cs.query(term))
        else:
            df = df_raw

        if self.idf_method == "lucene":
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        elif self.idf_method == "atire":
            idf = math.log(max(1e-12, num_docs / max(1.0, df)))
        else:
            inner = (num_docs - df + 0.5) / (df + 0.5)
            if not self.allow_negative_idf and self.clamp_idf_nonneg and inner < 1.0:
                inner = 1.0
            idf = math.log(inner)
        if self.clamp_idf_nonneg and not self.allow_negative_idf and idf < 0.0:
            return 0.0
        return idf

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float, str]]:
        if self.term_ptrs is None or self.post_docids is None or self.post_tfs is None or self.doc_len is None:
            raise RuntimeError("Index not ready. Call fit_iter() before searching.")

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        unique_terms = sorted(set(tokens))

        num_docs = len(self.doc_names)
        avg_dl = self.avg_dl
        k1 = self.k
        b = self.b

        scores_arr = np.zeros(num_docs, dtype=np.float32)

        for term in unique_terms:
            term_id = self.term_to_id.get(term)
            if term_id is None:
                continue
            start = int(self.term_ptrs[term_id])
            end = int(self.term_ptrs[term_id + 1])
            if start == end:
                continue

            df_raw = float(end - start)
            idf = self._idf(df_raw, num_docs, term)

            doc_ids = self.post_docids[start:end]
            term_freqs = self.post_tfs[start:end]
            doc_lengths = self.doc_len[doc_ids]
            denominator = term_freqs + k1 * (1.0 - b + b * (doc_lengths / max(1e-9, avg_dl)))
            increments = idf * (term_freqs * (k1 + 1.0)) / np.maximum(denominator, 1e-9)

            np.add.at(scores_arr, doc_ids, increments.astype(np.float32, copy=False))

        if k <= 0:
            k = 10

        if _NUMBA_OK:
            values, indices = _numba_sorted_top_k(scores_arr, int(k), sorted=True)
        else:
            if k >= len(scores_arr):
                indices = np.argsort(scores_arr)[::-1]
                values = scores_arr[indices]
            else:
                part = np.argpartition(scores_arr, -k)[-k:]
                sub = scores_arr[part]
                order = np.argsort(sub)[::-1]
                indices = part[order]
                values = sub[order]

        mask = values > 0.0
        values = values[mask]
        indices = indices[mask]

        return [
            (int(doc_id), float(score), self.doc_names[int(doc_id)])
            for doc_id, score in zip(indices[:k], values[:k])
        ]


def retrieve_in_batches(
    model: SketchBM25,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    *,
    batch_size: int,
    top_k: int,
    n_threads: int,
) -> Tuple[Dict[str, Dict[str, float]], List[float]]:
    """Retrieve ``top_k`` documents for the provided queries in batches."""

    results: Dict[str, Dict[str, float]] = {}
    latencies_ms: List[float] = []

    def _search(query: str) -> List[Tuple[int, float, str]]:
        t0 = time.perf_counter()
        hits = model.search(query, top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return hits

    executor: Optional[ThreadPoolExecutor]
    executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None

    try:
        for start, end in batched_indices(len(query_texts), batch_size):
            batch_ids = query_ids[start:end]
            batch_queries = query_texts[start:end]

            if executor is not None:
                futures = [executor.submit(_search, query) for query in batch_queries]
                batch_results = [future.result() for future in futures]
            else:
                batch_results = [_search(query) for query in batch_queries]

            for qid, hits in zip(batch_ids, batch_results):
                results[qid] = {
                    (doc_name if doc_name is not None else model.doc_names[doc_id]): float(score)
                    for doc_id, score, doc_name in hits
                }
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return results, latencies_ms


def main(
    *,
    save_dir: Path,
    result_dir: Path,
    n_threads: int,
    batch_size: int,
    top_k: int,
    index_workers: int,
    sidf_mode: str,
    idf_method: str,
    allow_negative_idf: bool,
    epsilon: float,
    delta: float,
    k1: float,
    b: float,
    min_token_len: int,
    stopwords: str | None,
    stemmer_name: str | None,
) -> Path:
    dataset = "msmarco"

    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), str(save_dir))

    if dataset == "cqadupstack":
        merge_cqa_dupstack(data_path)

    split = "dev"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    corpus_ids = list(corpus.keys())
    corpus_texts = [f"{item['title']} {item['text']}" for item in corpus.values()]
    num_docs = len(corpus_ids)

    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    stopwords_set, stopwords_label = resolve_stopwords(stopwords)

    normalized_stemmer = None if stemmer_name in {None, "none"} else stemmer_name

    tokenizer = Tokenizer(
        stopwords=stopwords_set,
        stemmer_name=normalized_stemmer,
        min_len=min_token_len,
    )

    model = SketchBM25(
        k=k1,
        b=b,
        workers=index_workers,
        tokenizer=tokenizer,
        sidf_mode=sidf_mode,
        epsilon=epsilon,
        delta=delta,
        idf_method=idf_method,
        allow_negative_idf=allow_negative_idf,
    )

    timer = Timer("[sketchbm25-msmarco]")

    index_event = timer.start("Index")
    model.fit_iter(zip(corpus_ids, corpus_texts))
    timer.stop(index_event, show=True, n_total=num_docs)

    retrieve_event = timer.start("Batch retrieve")
    start_time = time.perf_counter()
    results_dict, latencies_ms = retrieve_in_batches(
        model,
        query_ids,
        query_texts,
        batch_size=batch_size,
        top_k=top_k,
        n_threads=max(1, n_threads),
    )
    retrieval_time = time.perf_counter() - start_time
    timer.stop(retrieve_event, show=True, n_total=len(query_texts))

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_dict, [1, 10, 100, 1000])

    max_mem_gb = get_max_memory_usage("GB")
    qps = len(query_texts) / retrieval_time if retrieval_time > 0 else 0.0
    latencies_arr = np.asarray(latencies_ms, dtype=np.float32) if latencies_ms else np.asarray([], dtype=np.float32)
    latency_summary = {
        "count": int(latencies_arr.size),
        "mean_ms": float(latencies_arr.mean()) if latencies_arr.size else 0.0,
        "p50_ms": float(np.percentile(latencies_arr, 50)) if latencies_arr.size else 0.0,
        "p95_ms": float(np.percentile(latencies_arr, 95)) if latencies_arr.size else 0.0,
        "p99_ms": float(np.percentile(latencies_arr, 99)) if latencies_arr.size else 0.0,
    }

    print("=" * 50)
    print("Dataset:", dataset)
    print(f"Queries: {len(query_texts):,}")
    print(f"Docs: {num_docs:,}")
    print(f"Threads: {n_threads}")
    print(f"Batch size: {batch_size}")
    print(f"IDF method: {idf_method} | allow_negative_idf={allow_negative_idf}")
    print(f"Retrieval time: {retrieval_time:.2f} s ({qps:.2f} q/s)")
    if latency_summary["count"]:
        print(
            "Latency ms (mean/p95/p99): "
            f"{latency_summary['mean_ms']:.2f} / {latency_summary['p95_ms']:.2f} / {latency_summary['p99_ms']:.2f}"
        )
    print(f"Max memory usage: {max_mem_gb:.2f} GB")
    print("-" * 50)
    print(ndcg)
    print(recall)
    print("=" * 50)

    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = result_dir / f"sketchbm25-msmarco-{time.strftime('%Y%m%d-%H%M%S')}.json"

    save_dict = {
        "model": "sketchbm25",
        "dataset": dataset,
        "sidf_mode": sidf_mode,
        "idf_method": idf_method,
        "allow_negative_idf": allow_negative_idf,
        "epsilon": epsilon,
        "delta": delta,
        "k1": k1,
        "b": b,
        "min_token_len": min_token_len,
        "stopwords": stopwords_label,
        "stemmer": stemmer_name or "none",
        "n_threads": n_threads,
        "batch_size": batch_size,
        "top_k": top_k,
        "index_workers": index_workers,
        "retrieval_time": retrieval_time,
        "queries_per_second": qps,
        "max_mem_gb": max_mem_gb,
        "latency_ms": latency_summary,
        "timing": timer.to_dict(underscore=True, lowercase=True),
        "stats": {
            "num_docs": num_docs,
            "num_queries": len(query_texts),
        },
        "scores": {
            "ndcg": {k: float(v) for k, v in ndcg.items()},
            "map": {k: float(v) for k, v in _map.items()},
            "recall": {k: float(v) for k, v in recall.items()},
            "precision": {k: float(v) for k, v in precision.items()},
        },
    }

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-threaded SketchBM25 retrieval on MS MARCO in batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save_dir", type=Path, default=Path("datasets"))
    parser.add_argument("--result_dir", type=Path, default=Path("results/sketchbm25"))
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--index_workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--sidf_mode", type=str, default="none", choices=["none", "cms", "cs"])
    parser.add_argument(
        "--idf_method",
        type=str,
        default="robertson",
        choices=["robertson", "lucene", "atire"],
        help="IDF formulation to use for scoring",
    )
    parser.add_argument(
        "--allow_negative_idf",
        action="store_true",
        help="Allow negative IDF values instead of clamping them to zero.",
    )
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--min_token_len", type=int, default=2)
    stopword_choices = sorted(_STOPWORD_PRESETS.keys()) + ["none"]
    parser.add_argument("--stopwords", type=str, default="en", choices=stopword_choices)
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
        index_workers=args.index_workers,
        sidf_mode=args.sidf_mode,
        idf_method=args.idf_method,
        allow_negative_idf=args.allow_negative_idf,
        epsilon=args.epsilon,
        delta=args.delta,
        k1=args.k1,
        b=args.b,
        min_token_len=args.min_token_len,
        stopwords=args.stopwords,
        stemmer_name=args.stemmer_name,
    )

    print(f"Saved results to {save_path}")
