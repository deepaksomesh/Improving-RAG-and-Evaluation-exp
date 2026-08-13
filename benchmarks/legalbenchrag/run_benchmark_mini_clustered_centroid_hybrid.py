import argparse
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
import numpy as np
from sklearn.cluster import KMeans
import warnings

os.environ["SAC_CACHE_DIR"] = str(pathlib.Path.cwd() / "data" / "cache_mini")
os.environ["SAC_SUMMARIES_DIR"] = str(pathlib.Path.cwd() / "data" / "summary_mini")
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

from sac_rag.data_models import Benchmark, Document, QAGroundTruth, RetrievalMethod, RetrievedSnippet
from sac_rag.utils.credentials import credentials
from sac_rag.utils.config_loader import load_strategy_from_file
from sac_rag.utils.retriever_factory import create_retriever
from sac_rag.methods.baseline import BaselineRetrievalStrategy
from sac_rag.methods.hybrid import HybridStrategy, HybridRetrievalMethod
from sac_rag.utils.utils import sanitize_filename
from sac_rag.utils.stats_tracker import stats_tracker

# Import models from the baseline script for evaluation
from run_benchmark_mini_centroid import QAResult, BenchmarkResult, run_strategy, setup_and_load_data, benchmark_name_to_weight, create_summary_row

from sac_rag.utils.chunking import get_chunks, Chunk
from sac_rag.utils.ai import ai_embedding, AIEmbeddingType
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever

class ClusteredCentroidHybridRetrievalMethod(HybridRetrievalMethod):
    def __init__(self, retrieval_strategy, alpha=1.0):
        # HybridRetrievalMethod does not take cache_dir in its __init__, it relies on global env or hardcoded path.
        super().__init__(retrieval_strategy)
        self.alpha = alpha

    def _get_unique_collection_name(self, strategy) -> str:
        name = super()._get_unique_collection_name(strategy)
        return name.replace("..", "_")

    def _get_llama_embed_model(self):
        try:
            return super()._get_llama_embed_model()
        except ValueError:
            from llama_index.core.embeddings import MockEmbedding
            print("HybridClusteredCentroid: Bypassing LlamaIndex model with MockEmbedding since we provide vectors directly.")
            return MockEmbedding(embed_dim=768)

    async def sync_all_documents(self) -> None:
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"HybridClusteredCentroid (alpha={self.alpha}): Calculating chunks...")

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
            chunking_params["use_cache"] = self.retrieval_strategy.chunking_strategy.use_cache

        all_chunks_tasks = []
        for document in self.documents.values():
            task = asyncio.create_task(get_chunks(document=document, **chunking_params))
            all_chunks_tasks.append(task)

        results_of_chunking_tasks = await asyncio.gather(*all_chunks_tasks)
        all_chunks: List[Chunk] = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        stats_tracker.set('chunks_created', len(all_chunks))
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"HybridClusteredCentroid: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            print("HybridClusteredCentroid: No chunks created, skipping index creation.")
            return

        stats_tracker.start_timer('embedding_generation')
        chunk_contents = [chunk.content for chunk in all_chunks]
        model_config = self.retrieval_strategy.embedding_model

        pbar = tqdm(total=len(chunk_contents), desc="Hybrid Embeddings", ncols=100)
        def progress_callback(): pbar.update(1)

        embeddings = await ai_embedding(
            model=model_config,
            texts=chunk_contents,
            embedding_type=AIEmbeddingType.DOCUMENT,
            callback=progress_callback,
        )
        pbar.close()
        stats_tracker.stop_timer('embedding_generation')

        if len(embeddings) != len(all_chunks):
            raise ValueError(f"Hybrid Error: Mismatch between number of chunks and obtained embeddings.")

        # === CLUSTERED CENTROID ANCHORING LOGIC ===
        print(f"HybridClusteredCentroid: Performing centroid anchoring with alpha={self.alpha}...")
        emb_np = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb_np = emb_np / norms

        doc_indices = {}
        for i, chunk in enumerate(all_chunks):
            if chunk.file_path not in doc_indices:
                doc_indices[chunk.file_path] = []
            doc_indices[chunk.file_path].append(i)

        centroids = np.zeros_like(emb_np)
        for doc_id, indices in doc_indices.items():
            doc_embs = emb_np[indices]
            n_chunks = len(doc_embs)
            k = max(1, n_chunks // 5)
            
            if k == 1 or n_chunks < 2:
                centroid = np.mean(doc_embs, axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 0:
                    centroid = centroid / c_norm
                centroids[indices] = centroid
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(doc_embs)
                cluster_centers = kmeans.cluster_centers_
                c_norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
                c_norms[c_norms == 0] = 1e-10
                cluster_centers = cluster_centers / c_norms
                for local_idx, global_idx in enumerate(indices):
                    cluster_label = labels[local_idx]
                    centroids[global_idx] = cluster_centers[cluster_label]
            
        anchored_np = self.alpha * emb_np + (1.0 - self.alpha) * centroids
        a_norms = np.linalg.norm(anchored_np, axis=1, keepdims=True)
        a_norms[a_norms == 0] = 1e-10
        anchored_np = anchored_np / a_norms
        final_embeddings = anchored_np.tolist()
        # ==========================================

        print("HybridClusteredCentroid: Ingesting nodes into ChromaDB in batches...")
        batch_size = 512
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Ingesting to ChromaDB"):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_embeddings = final_embeddings[i:i + batch_size]
            batch_nodes = []

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                node_id = f"{chunk.file_path}_{chunk.span[0]}_{chunk.span[1]}"
                node = TextNode(
                    id_=node_id,
                    text=chunk.content,
                    metadata={
                        "file_path": chunk.file_path,
                        "original_span_start": chunk.span[0],
                        "original_span_end": chunk.span[1],
                    },
                    embedding=embedding,
                )
                batch_nodes.append(node)

            await self.vector_store.async_add(batch_nodes)
            self._bm25_nodes.extend(batch_nodes)

        print(f"HybridClusteredCentroid: Finished ingesting {len(all_chunks)} nodes into ChromaDB.")

        embed_model = self._get_llama_embed_model()
        self.vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=embed_model
        )

        print("HybridClusteredCentroid: Building BM25 retriever...")
        if not self._bm25_nodes:
            self.bm25_retriever = None
        else:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self._bm25_nodes,
                similarity_top_k=self.retrieval_strategy.bm25_top_k
            )


async def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    stats_tracker.start_timer('overall_run')
    start_time = datetime.now()
    print(f"Starting Legalbench-RAG Hybrid Clustered Centroid benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()
    os.environ["VOYAGEAI_API_KEY"] = credentials.ai.voyageai_api_key.get_secret_value()

    corpus, tests, weights = setup_and_load_data(args.max_tests_per_benchmark, args.sort_by_document, args.seed)
    
    run_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    alpha_values = [0.25]
    for alpha in alpha_values:
        alpha_results_dir = os.path.join(results_dir, f"alpha_{alpha}")
        os.makedirs(alpha_results_dir, exist_ok=True)
        summary_rows = []

        for i, config_path in enumerate(args.retrieval_configs):
            try:
                strategy = load_strategy_from_file(config_path)
                
                # FORCE HYBRID STRATEGY
                if getattr(strategy, 'strategy_type', None) != 'hybrid' and not isinstance(strategy, HybridStrategy):
                    print(f"WARNING: Forcing strategy type to hybrid")
                    # Dynamically convert to HybridStrategy if it's Baseline
                    hybrid_dict = strategy.model_dump()
                    hybrid_dict['strategy_type'] = 'hybrid'
                    hybrid_dict['bm25_top_k'] = 64
                    hybrid_dict['fusion_top_k'] = 64
                    hybrid_dict['fusion_weight'] = 0.5
                    strategy = HybridStrategy(**hybrid_dict)

                class CurrentCentroidMethod(ClusteredCentroidHybridRetrievalMethod):
                    def __init__(self, retrieval_strategy):
                        super().__init__(retrieval_strategy, alpha=alpha)
                
                import sac_rag.utils.retriever_factory
                sac_rag.utils.retriever_factory.HybridRetrievalMethod = CurrentCentroidMethod

                retriever = create_retriever(strategy)
                results_by_k = await run_strategy(tests, corpus, retriever, strat=strategy, weights=weights)

                for k, result in results_by_k.items():
                    config_basename, _ = os.path.splitext(os.path.basename(config_path))
                    result_filename = os.path.join(alpha_results_dir, f"{i}_{config_basename}_k{k}.json")
                    with open(result_filename, "w", encoding='utf-8') as f:
                        f.write(result.model_dump_json(indent=2))
                    row = create_summary_row(i, config_path, strategy, result, k)
                    summary_rows.append(row)
            except Exception as e:
                import traceback
                traceback.print_exc()

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(alpha_results_dir, "results_summary.csv")
            df.to_csv(summary_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LegalBench-RAG-mini Hybrid Clustered Centroid Evaluation")
    parser.add_argument("--retrieval-configs", "-rc", nargs='+', required=True)
    parser.add_argument("--max-tests-per-benchmark", "-m", type=int, default=194)
    parser.add_argument("--sort-by-document", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="./results/legalbenchrag_clustered_centroid_hybrid")
    args = parser.parse_args()
    asyncio.run(main(args))
