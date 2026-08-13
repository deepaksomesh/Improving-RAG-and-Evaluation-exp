import argparse
import asyncio
import logging
import math
import os
import random
from typing import List, Tuple, Dict, Any
import pandas as pd
from pydantic import BaseModel, computed_field, model_validator
from functools import cached_property
from tqdm import tqdm
from typing_extensions import Self
from datetime import datetime
import re

from sac_rag.data_models import Benchmark, Document, QAGroundTruth, RetrievalMethod, RetrievedSnippet
from sac_rag.utils.credentials import credentials
from sac_rag.utils.hyde import generate_legal_hypothesis
from sac_rag.utils.config_loader import load_strategy_from_file
from sac_rag.utils.retriever_factory import create_retriever
from sac_rag.methods.baseline import BaselineRetrievalStrategy, BaselineRetrievalMethod, serialize_f32, EmbeddingInfo
from sac_rag.methods.hybrid import HybridStrategy
from sac_rag.utils.utils import sanitize_filename
from sac_rag.utils.stats_tracker import stats_tracker

# Add imports needed for DocNameBaselineRetrievalMethod and CRUX
from sac_rag.utils.chunking import get_chunks
from sac_rag.utils.ai import ai_embedding, AIEmbeddingType, set_ai_cache_name
import sqlite3
import sqlite_vec
import sac_rag.utils.retriever_factory

class DocNameBaselineRetrievalMethod(BaselineRetrievalMethod):
    async def sync_all_documents(self) -> None:
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"DocNameBaseline: Calculating chunks using strategy '{self.retrieval_strategy.chunking_strategy.strategy_name}'...")

        chunking_params = {
            "strategy_name": self.retrieval_strategy.chunking_strategy.strategy_name,
            "chunk_size": self.retrieval_strategy.chunking_strategy.chunk_size,
            "chunk_overlap_ratio": self.retrieval_strategy.chunking_strategy.chunk_overlap_ratio,
        }
        if self.retrieval_strategy.chunking_strategy.strategy_name.startswith("summary_"):
            chunking_params["summarization_model"] = self.retrieval_strategy.chunking_strategy.summary_model
            chunking_params["summary_prompt_template"] = self.retrieval_strategy.chunking_strategy.summary_prompt_template
            chunking_params["prompt_target_char_length"] = self.retrieval_strategy.chunking_strategy.prompt_target_char_length
            chunking_params["summary_truncation_length"] = self.retrieval_strategy.chunking_strategy.summary_truncation_length
            chunking_params["use_cache"] = getattr(self.retrieval_strategy.chunking_strategy, 'use_cache', True)

        all_chunks_tasks = []
        for document in self.documents.values():
            task = asyncio.create_task(get_chunks(document=document, **chunking_params))
            all_chunks_tasks.append(task)

        results_of_chunking_tasks = await asyncio.gather(*all_chunks_tasks)

        all_chunks = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        stats_tracker.set('chunks_created', len(all_chunks))
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"DocNameBaseline: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            self.embedding_infos = []
            return

        self.embedding_infos = []
        stats_tracker.start_timer('embedding_generation')
        progress_bar = tqdm(total=len(all_chunks), desc="DocNameBaseline: Processing Embeddings", ncols=100)

        def progress_callback():
            if progress_bar:
                progress_bar.update(1)

        # ====== THE DOCNAME INJECTION ======
        chunk_contents = [
            f"Document Name: {os.path.basename(chunk.file_path)}\n\n{chunk.content}" 
            for chunk in all_chunks
        ]
        # ===================================

        embeddings = await ai_embedding(
            self.retrieval_strategy.embedding_model,
            chunk_contents,
            AIEmbeddingType.DOCUMENT,
            callback=progress_callback,
        )

        if progress_bar:
            progress_bar.close()
        stats_tracker.stop_timer('embedding_generation')

        print(f"DocNameBaseline: Start indexing embeddings into SQLite...")

        if self.sqlite_db is None:
            if os.path.exists(self.sqlite_db_file_path):
                import time
                for _ in range(10):
                    try:
                        os.remove(self.sqlite_db_file_path)
                        break
                    except PermissionError:
                        time.sleep(0.5)
                else:
                    os.remove(self.sqlite_db_file_path)
            self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path, check_same_thread=False)
            self.sqlite_db.enable_load_extension(True)
            sqlite_vec.load(self.sqlite_db)
            self.sqlite_db.enable_load_extension(False)
            self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
            if embeddings:
                self.sqlite_db.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                )

        if not embeddings:
            return

        with self.sqlite_db:
            for i, emb in enumerate(embeddings):
                self.sqlite_db.execute(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    (i + 1, serialize_f32(emb)),
                )
                chunk = all_chunks[i]
                self.embedding_infos.append(
                    EmbeddingInfo(
                        document_id=chunk.file_path,
                        span=chunk.span,
                        processed_content=chunk_contents[i],
                    )
                )

        print(f"DocNameBaseline: Finished indexing {len(self.embedding_infos)} embeddings.")

sac_rag.utils.retriever_factory.BaselineRetrievalMethod = DocNameBaselineRetrievalMethod


# --- Pydantic Models for this Benchmark's Evaluation Logic ---
class QAResult(BaseModel):
    qa_gt: QAGroundTruth
    retrieved_snippets: list[RetrievedSnippet]
    crux_coverage: float = 0.0
    crux_alpha_ndcg: float = 0.0
    crux_density: float = 0.0

    @cached_property
    def _relevant_retrieved_length(self) -> int:
        """Calculates the total character overlap. Is cached after the first call."""
        overlap_len = 0
        for snippet in self.retrieved_snippets:
            for gt_snippet in self.qa_gt.snippets:
                if snippet.file_path == gt_snippet.file_path:
                    # Calculate the length of the overlapping segment
                    overlap_start = max(snippet.span[0], gt_snippet.span[0])
                    overlap_end = min(snippet.span[1], gt_snippet.span[1])
                    if overlap_end > overlap_start:
                        overlap_len += overlap_end - overlap_start
        return overlap_len

    @cached_property
    def _total_retrieved_length(self) -> int:
        """Calculates the total length of all retrieved snippets."""
        return sum(s.span[1] - s.span[0] for s in self.retrieved_snippets)

    @cached_property
    def _total_relevant_length(self) -> int:
        """Calculates the total length of all ground truth snippets."""
        return sum(gt.span[1] - gt.span[0] for gt in self.qa_gt.snippets)

    # --- Public API Properties (now simple and clean) ---

    @computed_field
    @property
    def precision(self) -> float:
        if self._total_retrieved_length == 0:
            # If nothing was retrieved, precision is conventionally 0 or 1.
            # 0 is safer as it won't inflate scores for retrievers that return nothing.
            return 0.0
        return self._relevant_retrieved_length / self._total_retrieved_length

    @computed_field
    @property
    def recall(self) -> float:
        if self._total_relevant_length == 0:
            # This case should be rare, but if there's no ground truth text, recall is undefined or 1.
            # Returning 0.0 is a safe default.
            return 0.0
        return self._relevant_retrieved_length / self._total_relevant_length


def avg(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else float("nan")


class BenchmarkResult(BaseModel):
    qa_result_list: list[QAResult]
    weights: list[float]

    def get_avg_recall_and_precision(self, tag_filter: str | None = None) -> tuple[float, float]:
        indices = [
            i for i, qa_result in enumerate(self.qa_result_list)
            if tag_filter is None or tag_filter in qa_result.qa_gt.tags
        ]
        if not indices:
            return float("nan"), float("nan")

        filtered_results = [self.qa_result_list[i] for i in indices]
        filtered_weights = [self.weights[i] for i in indices]

        total_weight = sum(filtered_weights)
        if total_weight == 0:  # Unweighted average
            avg_recall = avg([r.recall for r in filtered_results])
            avg_precision = avg([r.precision for r in filtered_results])
            return avg_recall, avg_precision

        # Weighted average
        recall_weighted_avg = sum(r.recall * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        precision_weighted_avg = sum(r.precision * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        return recall_weighted_avg, precision_weighted_avg

    def get_avg_crux_metrics(self, tag_filter: str | None = None) -> tuple[float, float, float]:
        indices = [
            i for i, qa_result in enumerate(self.qa_result_list)
            if tag_filter is None or tag_filter in qa_result.qa_gt.tags
        ]
        if not indices:
            return float("nan"), float("nan"), float("nan")

        filtered_results = [self.qa_result_list[i] for i in indices]
        filtered_weights = [self.weights[i] for i in indices]

        total_weight = sum(filtered_weights)
        if total_weight == 0:
            avg_cov = avg([r.crux_coverage for r in filtered_results])
            avg_ndcg = avg([r.crux_alpha_ndcg for r in filtered_results])
            avg_den = avg([r.crux_density for r in filtered_results])
            return avg_cov, avg_ndcg, avg_den

        cov_w = sum(r.crux_coverage * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        ndcg_w = sum(r.crux_alpha_ndcg * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        den_w = sum(r.crux_density * w for r, w in zip(filtered_results, filtered_weights)) / total_weight
        return cov_w, ndcg_w, den_w

    @computed_field
    @property
    def avg_precision(self) -> float:
        return self.get_avg_recall_and_precision()[1]

    @computed_field
    @property
    def avg_recall(self) -> float:
        return self.get_avg_recall_and_precision()[0]

    @computed_field
    @property
    def avg_f1_score(self) -> float:
        precision, recall = self.avg_precision, self.avg_recall
        if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall > 0):
            return 2 * (precision * recall) / (precision + recall)
        return float('nan')

    @computed_field
    @property
    def avg_crux_coverage(self) -> float:
        return self.get_avg_crux_metrics()[0]

    @computed_field
    @property
    def avg_crux_alpha_ndcg(self) -> float:
        return self.get_avg_crux_metrics()[1]

    @computed_field
    @property
    def avg_crux_density(self) -> float:
        return self.get_avg_crux_metrics()[2]

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if len(self.qa_result_list) != len(self.weights):
            raise ValueError("Length of qa_result_list and weights do not match!")
        return self


# --- CRUX Evaluation Logic ---

def evaluate_crux_deterministic(qa_gt, retrieved_snippets, k_values):
    crux_metrics_by_k = {}
    alpha = 0.5
    
    gt_snippets = qa_gt.snippets
    if not gt_snippets:
        return {k: {"coverage": 0.0, "alpha_ndcg": 0.0, "density": 0.0} for k in k_values}
        
    for k in k_values:
        if k == 0:
            crux_metrics_by_k[k] = {"coverage": 0.0, "alpha_ndcg": 0.0, "density": 0.0}
            continue
            
        top_retrieved = retrieved_snippets[:k]
        
        covered_gt_indices = set()
        useful_chunk_count = 0
        
        for chunk in top_retrieved:
            chunk_is_useful = False
            for gt_idx, gt in enumerate(gt_snippets):
                if chunk.file_path == gt.file_path:
                    overlap_start = max(chunk.span[0], gt.span[0])
                    overlap_end = min(chunk.span[1], gt.span[1])
                    if overlap_end > overlap_start:
                        covered_gt_indices.add(gt_idx)
                        chunk_is_useful = True
            if chunk_is_useful:
                useful_chunk_count += 1
                
        coverage = len(covered_gt_indices) / len(gt_snippets)
        density = useful_chunk_count / k if k > 0 else 0.0
        
        import math
        idcg = 0.0
        for r in range(min(k, len(gt_snippets))):
            idcg += 1.0 / math.log2(r + 1 + 1)
            
        dcg = 0.0
        c_i = [0] * len(gt_snippets)
        for r_idx, chunk in enumerate(top_retrieved):
            ng = 0.0
            for gt_idx, gt in enumerate(gt_snippets):
                I = 0
                if chunk.file_path == gt.file_path:
                    overlap_start = max(chunk.span[0], gt.span[0])
                    overlap_end = min(chunk.span[1], gt.span[1])
                    if overlap_end > overlap_start:
                        I = 1
                ng += I * ((1 - alpha) ** c_i[gt_idx])
                c_i[gt_idx] += I
            dcg += ng / math.log2(r_idx + 1 + 1)
            
        alpha_ndcg = dcg / idcg if idcg > 0 else 0.0
        
        crux_metrics_by_k[k] = {
            "coverage": coverage,
            "alpha_ndcg": alpha_ndcg,
            "density": density
        }
        
    return crux_metrics_by_k


# --- Core Benchmark Execution Logic ---

async def run_strategy(
        qa_gt_list: list[QAGroundTruth],
        corpus: list[Document],
        retriever: RetrievalMethod,
        strat: Any,
        weights: list[float] | None = None,
) -> Dict[int, BenchmarkResult]:
    """Executes a benchmark run for a given retriever and test set for multiple top-k-values."""
    for document in tqdm(corpus, desc="Ingesting documents"):
        await retriever.ingest_document(document)
    await retriever.sync_all_documents()

    final_k_values = strat.rerank_top_k
    if not final_k_values:
        # Pydantic checks that rerank_top_k is a List[int] so this should not happen
        raise ValueError("No 'rerank_top_k' list found in the strategy config.")

    # This will store the full, un-truncated results for each query
    full_query_results: List[Tuple[QAGroundTruth, List[RetrievedSnippet], Dict[int, Dict[str, float]]]] = []
    
    # 2. Run Queries
    print(f"Executing {len(qa_gt_list)} queries and evaluating CRUX...")
    stats_tracker.start_timer('query_processing')

    query_semaphore = asyncio.Semaphore(70)

    async def run_query(qa_gt: QAGroundTruth) -> Tuple[QAGroundTruth, List[RetrievedSnippet], Dict[int, Dict[str, float]]]:
        async with query_semaphore:
            dataset_name = "privacy_qa"
            valid_datasets = ["privacy_qa", "maud", "cuad", "contractnli"]
            for tag in qa_gt.tags:
                if tag in valid_datasets:
                    dataset_name = tag
                    break
            
            try:
                # Legal-HyDE Generation
                hypothesis = await generate_legal_hypothesis(qa_gt.query, dataset_name)
                
                # Search the vector DB using the combined original query and hypothetical clause.
                combined_query = f"{qa_gt.query}\n\n{hypothesis}"
                query_response = await retriever.query(combined_query)
                retrieved_snippets = query_response.retrieved_snippets
            except Exception as e:
                print(f"Error executing query '{qa_gt.query[:50]}...': {e}")
                retrieved_snippets = []
                
        try:
            crux_metrics = evaluate_crux_deterministic(qa_gt, retrieved_snippets, final_k_values)
        except Exception as e:
            print(f"Error evaluating CRUX for query: {e}")
            crux_metrics = {k: {"coverage": 0.0, "alpha_ndcg": 0.0, "density": 0.0} for k in final_k_values}
            
        return qa_gt, retrieved_snippets, crux_metrics

    tasks = [run_query(qa_gt) for qa_gt in qa_gt_list]
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Running queries"):
        result = await future
        full_query_results.append(result)

    await retriever.cleanup()

    stats_tracker.stop_timer('query_processing')

    # Post-process results for each top-k
    results_by_k: Dict[int, BenchmarkResult] = {}
    for k in final_k_values:
        qa_results_for_k: list[QAResult] = []
        for qa_gt, full_retrieved_snippets, crux_metrics in full_query_results:
            # Slice the results for the current top-k
            sliced_snippets = full_retrieved_snippets[:k]
            metrics_for_k = crux_metrics.get(k, {"coverage": 0.0, "alpha_ndcg": 0.0, "density": 0.0})
            qa_results_for_k.append(
                QAResult(
                    qa_gt=qa_gt, 
                    retrieved_snippets=sliced_snippets,
                    crux_coverage=metrics_for_k["coverage"],
                    crux_alpha_ndcg=metrics_for_k["alpha_ndcg"],
                    crux_density=metrics_for_k["density"]
                )
            )

        results_by_k[k] = BenchmarkResult(
            qa_result_list=qa_results_for_k,
            weights=weights if weights is not None else [1.0] * len(qa_results_for_k),
        )

    return results_by_k


# --- Data Setup Logic ---

def setup_and_load_data(max_tests: int, sort_by_doc: bool, seed: int, use_mini_dataset: bool) -> Tuple[List[Document], List[QAGroundTruth], List[float]]:
    """Loads, samples, and prepares all data needed for the benchmark."""
    all_tests, weights, used_doc_paths = [], [], set()

    base_dir = "./legalbenchragmini" if use_mini_dataset else "./data"
    bench_dir = f"{base_dir}/benchmark" if use_mini_dataset else f"{base_dir}/benchmarks"

    for dataset_name, weight in benchmark_name_to_weight.items():
        benchmark_file = f"{bench_dir}/{dataset_name}.json"
        if not os.path.exists(benchmark_file):
            print(f"Warning: Benchmark file not found: {benchmark_file}. Skipping.")
            continue

        # Load benchmark QA including ground truth snippets
        with open(benchmark_file, encoding='utf-8') as f:
            tests = Benchmark.model_validate_json(f.read()).tests

        # Sanitize all snippet file paths in place, immediately after loading.
        # This ensures all subsequent logic uses the canonical, sanitized path.
        for test in tests:
            ignore_prefix = f"{dataset_name}/"  # "maud/"  # TODO: f"{dataset_name}/"
            for snippet in test.snippets:
                snippet.orig_file_path = snippet.file_path  # Store the original raw path

                # Original: "dataset/file/with\\slashes.txt"
                # Sanitized: "dataset/file_with_slashes.txt"
                snippet.file_path = sanitize_filename(snippet.file_path, ignore_dirs=ignore_prefix)  # Path on disk

        # Sampling logic
        sampled_tests = tests
        if 0 < max_tests < len(tests):
            print(f"Sampling {max_tests} tests from {dataset_name} ({len(tests)} total)")
            if sort_by_doc:
                tests = sorted(tests, key=lambda t: t.snippets[0].file_path if t.snippets else "")
            else:
                if seed is not None:
                    random.seed(seed)
                else:
                    random.seed(dataset_name + str(max_tests))
                random.shuffle(tests)
            sampled_tests = tests[:max_tests]

        for t in sampled_tests:
            for s in t.snippets:
                used_doc_paths.add(s.file_path)

        for t in sampled_tests:
            t.tags = [dataset_name]

        all_tests.extend(sampled_tests)
        if sampled_tests:
            per_test_weight = weight / len(sampled_tests)
            weights.extend([per_test_weight] * len(sampled_tests))

    print(f"Total tests selected across all benchmarks: {len(all_tests)}")

    # Corpus loading
    corpus, loaded_paths = [], set()
    print(f"Attempting to load {len(used_doc_paths)} required corpus documents...")
    for doc_path in sorted(list(used_doc_paths)):
        full_path = f"{base_dir}/corpus/{doc_path}"
        if not os.path.exists(full_path):
            print(f"Warning: Corpus file not found at '{full_path}'. Skipping.")
            continue

        with open(full_path, encoding='utf-8') as f:
            content = f.read()
            if content.strip():
                corpus.append(Document(file_path=doc_path, content=content))
                loaded_paths.add(doc_path)

    print(f"Successfully loaded {len(loaded_paths)} corpus documents.")

    # Filter tests to only those with loaded documents
    final_tests, final_weights = [], []
    for i, test in enumerate(all_tests):
        all_loaded = True
        for s in test.snippets:
            if s.file_path not in loaded_paths:
                all_loaded = False
                break
        if all_loaded:
            final_tests.append(test)
            final_weights.append(weights[i])

    if len(final_tests) != len(all_tests):  # should never happen
        print(f"Filtered out {len(all_tests) - len(final_tests)} tests due to missing corpus files.")

    if not final_tests:
        raise RuntimeError("No valid tests remaining after document filtering. Exiting.")

    return corpus, final_tests, final_weights


benchmark_name_to_weight: dict[str, float] = {
    "privacy_qa": 0.25, "contractnli": 0.25, "maud": 0.25, "cuad": 0.25
}

benchmark_name_to_weight_test: dict[str, float] = {
    "lbrag_test": 1.0
}


def create_summary_row(idx: int, config_path: str, strategy: Any, result: BenchmarkResult, top_k: int) -> Dict[str, Any]:
    """Creates a detailed dictionary row for the summary CSV."""

    # Start with basic info
    row = {
        "i": idx,
        "config_file": config_path,
        "recall": result.avg_recall,
        "precision": result.avg_precision,
        "f1_score": result.avg_f1_score,
        "crux_coverage": result.avg_crux_coverage,
        "crux_alpha_ndcg": result.avg_crux_alpha_ndcg,
        "crux_density": result.avg_crux_density,
        "chunk_strategy_name": strategy.chunking_strategy.strategy_name,
        "chunk_size": strategy.chunking_strategy.chunk_size,
        "embedding_model_company": strategy.embedding_model.company,
        "embedding_model_name": strategy.embedding_model.model,
        "embedding_top_k": strategy.embedding_top_k,
        "rerank_model_company": strategy.rerank_model.company if strategy.rerank_model else None,
        "rerank_model_name": strategy.rerank_model.model if strategy.rerank_model else None,
        "rerank_top_k": top_k,
    }

    # Deconstruct the strategy object to get detailed columns
    if isinstance(strategy, BaselineRetrievalStrategy):
        row.update({
            "method": "baseline",
        })
    elif isinstance(strategy, HybridStrategy):
        row.update({
            "method": "hybrid",
            "bm25_top_k": strategy.bm25_top_k,
            "fusion_top_k": strategy.fusion_top_k,
            "fusion_weight": strategy.fusion_weight,
        })
    else:
        print("WARNING: Unsupported strategy type. Skipping.")

    # Add the per-benchmark metrics
    for benchmark_name in benchmark_name_to_weight:
        avg_recall, avg_precision = result.get_avg_recall_and_precision(benchmark_name)
        avg_cov, avg_ndcg, avg_den = result.get_avg_crux_metrics(benchmark_name)
        row[f"{benchmark_name}|recall"] = avg_recall
        row[f"{benchmark_name}|precision"] = avg_precision
        row[f"{benchmark_name}|crux_coverage"] = avg_cov
        row[f"{benchmark_name}|crux_alpha_ndcg"] = avg_ndcg
        row[f"{benchmark_name}|crux_density"] = avg_den

        f1 = float('nan')
        if not (math.isnan(avg_precision) or math.isnan(avg_recall)) and (avg_precision + avg_recall > 0):
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        row[f"{benchmark_name}|f1_score"] = f1

    return row


# --- Main Orchestrator ---

async def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # TODO: revert to logging.WARNING
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    if getattr(args, 'cache_name', None):
        set_ai_cache_name(args.cache_name)

    stats_tracker.start_timer('overall_run')
    start_time = datetime.now()
    print(f"Starting Legalbench-RAG DocName Baseline benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()
    os.environ["VOYAGEAI_API_KEY"] = credentials.ai.voyageai_api_key.get_secret_value()

    # 1. Setup and load all data once
    stats_tracker.start_timer('data_setup')
    corpus, tests, weights = setup_and_load_data(args.max_tests_per_benchmark, args.sort_by_document, args.seed, args.use_mini_dataset)
    stats_tracker.stop_timer('data_setup')
    stats_tracker.set('documents_processed', len(corpus))
    stats_tracker.set('queries_processed', len(tests))

    # 2. Prepare for results
    run_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Benchmark results will be saved to: {results_dir}")

    summary_rows = []

    # 3. Loop through each provided configuration file
    for i, config_path in enumerate(args.retrieval_configs):
        print(f"\n--- Running Config {i + 1}/{len(args.retrieval_configs)}: {config_path} ---")

        try:
            strategy = load_strategy_from_file(config_path)
            retriever = create_retriever(strategy, db_name=getattr(args, 'cache_name', None))

            # Execute the benchmark
            results_by_k = await run_strategy(tests, corpus, retriever, strat=strategy, weights=weights)

            # Loop through the results for each k and save/summarize
            for k, result in results_by_k.items():
                print(f"--- Post-Processing results for k={k} ---")

                # Save detailed JSON result for this run and top-k
                config_basename, _ = os.path.splitext(os.path.basename(config_path))
                result_filename = os.path.join(results_dir, f"{i}_{config_basename}_k{k}.json")
                with open(result_filename, "w", encoding='utf-8') as f:
                    f.write(result.model_dump_json(indent=2))

                # Prepare the DETAILED summary row
                row = create_summary_row(i, config_path, strategy, result, k)
                summary_rows.append(row)

                print(f"  Overall Avg Recall:    {100 * result.avg_recall: .2f}%")
                print(f"  Overall Avg Precision: {100 * result.avg_precision: .2f}%")
                print(f"  Overall Avg F1-Score:  {100 * result.avg_f1_score: .2f}%")
                print(f"  Overall CRUX Coverage: {result.avg_crux_coverage: .4f}")
                print(f"  Overall CRUX a-nDCG:   {result.avg_crux_alpha_ndcg: .4f}")
                print(f"  Overall CRUX Density:  {result.avg_crux_density: .4f}")

        except Exception as e:
            import traceback
            print(f"!!!!!!!!!!!! ERROR running benchmark for config {config_path} !!!!!!!!!!!!")
            print(f"Error: {e}")
            traceback.print_exc()
            summary_rows.append(
                {"config_file": config_path, "recall": "ERROR", "precision": "ERROR", "f1_score": "ERROR",
                 "crux_coverage": "ERROR", "crux_alpha_ndcg": "ERROR", "crux_density": "ERROR"})

    # 4. Save final summary CSV and STATS
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(results_dir, "results_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f'\nOverall Benchmark summary saved to: "{summary_path}"')

    stats_tracker.stop_timer('overall_run')
    stats_report_content = stats_tracker.report()
    stats_path = os.path.join(results_dir, "stats.txt")
    try:
        with open(stats_path, "w", encoding='utf-8') as f:
            f.write(stats_report_content)
            print(f'\nOperational stats saved to: "{stats_path}"')
    except Exception as e:
        print(f"Error saving stats report: {e}. Skipping...")
    print("\n" + stats_report_content)  # Also print to console for convenience

    print(f"\nBenchmark run '{run_name}' finished.")

    end_time = datetime.now()
    print(f"Run finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the legalbench-rag benchmark.")
    parser.add_argument(
        "--retrieval-configs", "-rc",
        nargs='+', required=True,
        help="One or more paths to retrieval strategy JSON config files."
    )
    parser.add_argument(
        "--max-tests-per-benchmark", "-m", type=int, default=-1,
        help="Maximum number of tests to sample from each sub-benchmark (e.g., cuad, maud). Set a low number for debug."
    )
    parser.add_argument(
        "--sort-by-document", action="store_true",
        help="Enable sorting by document to potentially speed up ingestion during testing."
    )
    parser.add_argument(
        "--seed", type=int, default=None
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results/legalbenchrag_docname_legal_hyde_crux",
        help="Base directory to save the output run folder."
    )
    parser.add_argument(
        "--cache-name", type=str, default=None,
        help="Cache name to use for embeddings and SQLite database."
    )
    parser.add_argument(
        "--use-mini-dataset", action="store_true",
        help="Use the legalbenchragmini dataset instead of the full dataset."
    )

    args = parser.parse_args()
    asyncio.run(main(args))
