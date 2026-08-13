import os

script_content = """import argparse
import asyncio
import logging
import math
import os
import pathlib
import random
from typing import List, Tuple, Dict, Any
import pandas as pd
from pydantic import BaseModel, computed_field, model_validator
from functools import cached_property
from tqdm import tqdm
from typing_extensions import Self
from datetime import datetime
import warnings

os.environ["SAC_CACHE_DIR"] = str(pathlib.Path.cwd() / "data" / "cache_mini")
os.environ["SAC_SUMMARIES_DIR"] = str(pathlib.Path.cwd() / "data" / "summary_mini")

from sac_rag.data_models import Benchmark, Document, QAGroundTruth, RetrievalMethod, RetrievedSnippet, QueryResponse
from sac_rag.utils.credentials import credentials
from sac_rag.utils.config_loader import load_strategy_from_file
from sac_rag.utils.retriever_factory import create_retriever
from sac_rag.methods.baseline import BaselineRetrievalStrategy
from sac_rag.utils.utils import sanitize_filename
from sac_rag.utils.stats_tracker import stats_tracker

# Import models from the baseline script for evaluation
from benchmarks.legalbenchrag.run_benchmark_mini_centroid import QAResult, BenchmarkResult, run_strategy, setup_and_load_data, benchmark_name_to_weight, create_summary_row
from sac_rag.utils.chunking import get_chunks, Chunk

class ColBERTRetrievalMethod(RetrievalMethod):
    def __init__(self, retrieval_strategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.all_chunks = []
        try:
            from ragatouille import RAGPretrainedModel
            print("Loading ColBERT model (colbert-ir/colbertv2.0)...")
            self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        except ImportError:
            raise ImportError("ragatouille is not installed. Please run: pip install ragatouille")

    async def ingest_document(self, document: Document) -> None:
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"ColBERT: Calculating chunks...")
        chunking_params = {
            "strategy_name": self.retrieval_strategy.chunking_strategy.strategy_name,
            "chunk_size": self.retrieval_strategy.chunking_strategy.chunk_size,
            "chunk_overlap_ratio": self.retrieval_strategy.chunking_strategy.chunk_overlap_ratio,
        }
        all_chunks_tasks = []
        for document in self.documents.values():
            task = asyncio.create_task(get_chunks(document=document, **chunking_params))
            all_chunks_tasks.append(task)

        results_of_chunking_tasks = await asyncio.gather(*all_chunks_tasks)
        all_chunks: List[Chunk] = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)
            
        self.all_chunks = all_chunks
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"ColBERT: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            return

        print("ColBERT: Indexing documents with ragatouille...")
        docs = [c.content for c in all_chunks]
        metadatas = [{"file_path": c.file_path, "start": c.span[0], "end": c.span[1]} for c in all_chunks]
        
        # We use a unique index name based on time to avoid caching collisions during sweeps
        index_name = f"legalbenchrag_colbert_{datetime.now().strftime('%H%M%S')}"
        
        # We index using ColBERT. split_documents=False because we already chunked.
        index_path = self.RAG.index(collection=docs, document_metadatas=metadatas, index_name=index_name, max_document_length=512, split_documents=False)
        print(f"ColBERT: Index created at {index_path}")
        from ragatouille import RAGPretrainedModel
        self.RAG = RAGPretrainedModel.from_index(index_path)

    async def query(self, query: str) -> QueryResponse:
        results = self.RAG.search(query, k=self.retrieval_strategy.embedding_top_k)
        
        retrieved_snippets = []
        for i, res in enumerate(results):
            meta = res.get('document_metadata', {})
            file_path = meta.get('file_path', '')
            span = (meta.get('start', 0), meta.get('end', 0))
            score = res.get('score', 0.0)
            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=file_path,
                    span=span,
                    score=float(score),
                    full_chunk_text=res.get('content', '')
                )
            )
        return QueryResponse(retrieved_snippets=retrieved_snippets)
        
    async def cleanup(self) -> None:
        self.documents = {}
        self.all_chunks = []


async def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stats_tracker.start_timer('overall_run')
    start_time = datetime.now()
    print(f"Starting Legalbench-RAG ColBERT benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    corpus, tests, weights = setup_and_load_data(args.max_tests_per_benchmark, args.sort_by_document, args.seed)
    
    run_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    summary_rows = []

    for i, config_path in enumerate(args.retrieval_configs):
        try:
            strategy = load_strategy_from_file(config_path)

            import sac_rag.utils.retriever_factory
            sac_rag.utils.retriever_factory.BaselineRetrievalMethod = ColBERTRetrievalMethod

            retriever = create_retriever(strategy)
            
            # Note: We aren't doing the alpha sweep here since ColBERT is entirely dense token-level late interaction.
            results_by_k = await run_strategy(tests, corpus, retriever, strat=strategy, weights=weights)

            for k, result in results_by_k.items():
                config_basename, _ = os.path.splitext(os.path.basename(config_path))
                result_filename = os.path.join(results_dir, f"{i}_{config_basename}_k{k}.json")
                with open(result_filename, "w", encoding='utf-8') as f:
                    f.write(result.model_dump_json(indent=2))
                row = create_summary_row(i, config_path, strategy, result, k)
                summary_rows.append(row)
        except Exception as e:
            import traceback
            traceback.print_exc()

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(results_dir, "results_summary.csv")
        df.to_csv(summary_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LegalBench-RAG-mini ColBERT Evaluation")
    parser.add_argument("--retrieval-configs", "-rc", nargs='+', required=True)
    parser.add_argument("--max-tests-per-benchmark", "-m", type=int, default=194)
    parser.add_argument("--sort-by-document", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="./results/legalbenchrag_colbert")
    args = parser.parse_args()
    asyncio.run(main(args))
"""

with open('d:/Thesis/Baselines/summary-augmented-chunking/benchmarks/legalbenchrag/run_benchmark_mini_colbert.py', 'w', encoding='utf-8') as f:
    f.write(script_content)
