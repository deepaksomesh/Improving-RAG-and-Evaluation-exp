import os

source_path = 'benchmarks/legalbenchrag/run_benchmark_mini.py'
dest_path = 'benchmarks/legalbenchrag/run_benchmark_mini_mmr.py'

with open(source_path, 'r', encoding='utf-8') as f:
    content = f.read()

parts = content.split("async def main(args):")
header_content = parts[0]

mmr_class_code = """
import numpy as np
import os
import asyncio
import sqlite3
import sqlite_vec
from typing import List
from tqdm import tqdm

from sac_rag.methods.baseline import BaselineRetrievalMethod, serialize_f32, EmbeddingInfo
from sac_rag.utils.chunking import get_chunks, Chunk
from sac_rag.utils.ai import ai_embedding, AIEmbeddingType
from sac_rag.data_models import QueryResponse, RetrievedSnippet

GLOBAL_RAW_EMBEDDINGS_CACHE = {}

def mmr_rerank(query_vec, cand_vecs, lambda_param=0.5, top_k=64):
    \"\"\"
    query_vec:  (d,) normalized query embedding
    cand_vecs:  (N, d) normalized candidate embeddings, in similarity order
    returns:    list of indices into cand_vecs, in MMR order
    \"\"\"
    # all vectors already L2-normalized, so dot product = cosine
    query_sim = cand_vecs @ query_vec        # (N,) relevance to query
    doc_sim   = cand_vecs @ cand_vecs.T       # (N,N) pairwise similarity

    selected = []
    remaining = list(range(len(cand_vecs)))

    # first pick = most relevant (MMR == baseline at k=1)
    if not remaining:
        return selected

    first = int(np.argmax(query_sim))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, -np.inf
        for d in remaining:
            redundancy = max(doc_sim[d][s] for s in selected)
            score = lambda_param * query_sim[d] - (1 - lambda_param) * redundancy
            if score > best_score:
                best_score, best_idx = score, d
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

class MMRDocNameRetrievalMethod(BaselineRetrievalMethod):
    def __init__(self, retrieval_strategy, cache_dir=None, lambda_param=0.5):
        super().__init__(retrieval_strategy, cache_dir)
        self.lambda_param = lambda_param
        self.indexed_vectors = []

    async def sync_all_documents(self) -> None:
        from sac_rag.utils.stats_tracker import stats_tracker
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"MMRDocName (lambda={self.lambda_param}): Calculating chunks using strategy '{self.retrieval_strategy.chunking_strategy.strategy_name}'...")

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

        all_chunks: List[Chunk] = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        stats_tracker.set('chunks_created', len(all_chunks))
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"MMRDocName: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            self.embedding_infos = []
            return

        self.embedding_infos = []
        self.indexed_vectors = []
        
        stats_tracker.start_timer('embedding_generation')
        progress_bar = tqdm(total=len(all_chunks), desc="MMRDocName: Processing Embeddings", ncols=100)
        def progress_callback():
            if progress_bar:
                progress_bar.update(1)

        # ====== THE DOCNAME INJECTION ======
        chunk_contents = [
            f"Document Name: {os.path.basename(chunk.file_path)}\\n\\n{chunk.content}" 
            for chunk in all_chunks
        ]
        # ===================================
        
        embeddings = []
        contents_to_embed = []
        indices_to_embed = []
        
        for i, content in enumerate(chunk_contents):
            if content in GLOBAL_RAW_EMBEDDINGS_CACHE:
                embeddings.append(GLOBAL_RAW_EMBEDDINGS_CACHE[content])
            else:
                embeddings.append(None)
                contents_to_embed.append(content)
                indices_to_embed.append(i)
                
        if contents_to_embed:
            new_embeddings = await ai_embedding(
                self.retrieval_strategy.embedding_model,
                contents_to_embed,
                AIEmbeddingType.DOCUMENT,
                callback=progress_callback,
            )
            for i, idx in enumerate(indices_to_embed):
                embeddings[idx] = new_embeddings[i]
                GLOBAL_RAW_EMBEDDINGS_CACHE[contents_to_embed[i]] = new_embeddings[i]

        if progress_bar:
            progress_bar.close()
        stats_tracker.stop_timer('embedding_generation')

        # L2 Normalize the final vectors and store in memory
        emb_np = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        emb_np = emb_np / norms
        self.indexed_vectors = emb_np.tolist()

        print(f"MMRDocName: Start indexing embeddings into SQLite...")

        if self.sqlite_db is None:
            if os.path.exists(self.sqlite_db_file_path):
                os.remove(self.sqlite_db_file_path)
            self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
            self.sqlite_db.enable_load_extension(True)
            sqlite_vec.load(self.sqlite_db)
            self.sqlite_db.enable_load_extension(False)
            self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
            if self.indexed_vectors:
                self.sqlite_db.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(self.indexed_vectors[0])}])"
                )

        if not self.indexed_vectors:
            return

        with self.sqlite_db:
            for i, emb in enumerate(self.indexed_vectors):
                self.sqlite_db.execute(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    (i + 1, serialize_f32(emb)),
                )
                chunk = all_chunks[i]
                self.embedding_infos.append(
                    EmbeddingInfo(
                        document_id=chunk.file_path,
                        span=chunk.span,
                        processed_content=chunk.content,
                    )
                )

        print(f"MMRDocName: Finished indexing {len(self.embedding_infos)} embeddings.")

    async def query(self, query: str) -> QueryResponse:
        import logging
        from sac_rag.methods.baseline import serialize_f32
        logger = logging.getLogger(__name__)
        
        if self.sqlite_db is None or self.embedding_infos is None or not self.indexed_vectors:
            logger.error("RetrievalMethod not properly synchronized.")
            raise ValueError("Sync documents before querying!")

        if not self.embedding_infos:
            return QueryResponse(retrieved_snippets=[])

        query_embedding_raw = (
            await ai_embedding(
                self.retrieval_strategy.embedding_model, [query], AIEmbeddingType.QUERY
            )
        )[0]
        
        # === QUERY NORMALIZATION ===
        q_np = np.array(query_embedding_raw, dtype=np.float32)
        q_norm = np.linalg.norm(q_np)
        if q_norm > 0:
            q_np = q_np / q_norm
        
        # Retrieve candidate pool (128 for MMR reranking)
        pool_size = 128
        rows = self.sqlite_db.execute(
            \"\"\"
            SELECT
                rowid,
                distance
            FROM vec_items
            WHERE embedding MATCH ?
            ORDER BY distance ASC
            LIMIT ?
            \"\"\",
            [serialize_f32(q_np.tolist()), pool_size],
        ).fetchall()

        retrieved_metadatas = []
        cand_vecs_list = []
        for row_id_sqlite, _ in rows:
            meta_index = int(row_id_sqlite) - 1
            if 0 <= meta_index < len(self.embedding_infos):
                retrieved_metadatas.append(self.embedding_infos[meta_index])
                cand_vecs_list.append(self.indexed_vectors[meta_index])
            else:
                logger.warning(
                    f"WARNING - Invalid rowid {row_id_sqlite} from SQLite query."
                )

        # Rerank with MMR
        if cand_vecs_list:
            cand_vecs = np.array(cand_vecs_list, dtype=np.float32)
            # Rerank all retrieved candidates
            order = mmr_rerank(q_np, cand_vecs, lambda_param=self.lambda_param, top_k=len(cand_vecs))
            final_metadatas = [retrieved_metadatas[i] for i in order]
        else:
            final_metadatas = []

        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets = []
        for i, metadata in enumerate(final_metadatas):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            
            span = metadata.span
            text_content = metadata.processed_content
            current_len = len(text_content)
            current_full_chunk_text = metadata.processed_content

            final_span = span
            if remaining_tokens is not None:
                if current_len > remaining_tokens:
                    final_span = (span[0], span[0] + remaining_tokens)
                remaining_tokens -= current_len

            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=metadata.document_id,
                    span=final_span,
                    score=1.0 / (i + 1),
                    full_chunk_text=current_full_chunk_text
                )
            )

        return QueryResponse(retrieved_snippets=retrieved_snippets)
"""

new_main_code = """async def main(args):
    import pandas as pd
    from sac_rag.utils.stats_tracker import stats_tracker
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    stats_tracker.start_timer('overall_run')
    start_time = datetime.now()
    print(f"Starting Legalbench-RAG MMR benchmark run at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()
    os.environ["VOYAGEAI_API_KEY"] = credentials.ai.voyageai_api_key.get_secret_value()

    # 1. Setup and load all data once
    stats_tracker.start_timer('data_setup')
    corpus, tests, weights = setup_and_load_data(args.max_tests_per_benchmark, args.sort_by_document, args.seed)
    stats_tracker.stop_timer('data_setup')
    stats_tracker.set('documents_processed', len(corpus))
    stats_tracker.set('queries_processed', len(tests))

    run_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Benchmark results will be saved to: {results_dir}")

    # Sweep lambda values
    lambda_values = [1.0, 0.7, 0.5, 0.3]

    for lambda_val in lambda_values:
        lambda_results_dir = os.path.join(results_dir, f"lambda_{lambda_val}")
        os.makedirs(lambda_results_dir, exist_ok=True)
        print(f"\\n=======================================================")
        print(f"Starting Sweep for Lambda = {lambda_val}")
        print(f"=======================================================")

        summary_rows = []

        for i, config_path in enumerate(args.retrieval_configs):
            print(f"\\n--- Running Config {i + 1}/{len(args.retrieval_configs)}: {config_path} ---")

            try:
                strategy = load_strategy_from_file(config_path)

                # Monkey-patch the retriever factory to use our custom class with the current lambda
                class CurrentMMRMethod(MMRDocNameRetrievalMethod):
                    def __init__(self, retrieval_strategy, cache_dir=None):
                        super().__init__(retrieval_strategy, cache_dir, lambda_param=lambda_val)
                
                import sac_rag.utils.retriever_factory
                sac_rag.utils.retriever_factory.BaselineRetrievalMethod = CurrentMMRMethod

                retriever = create_retriever(strategy)

                # Execute the benchmark
                results_by_k = await run_strategy(tests, corpus, retriever, strat=strategy, weights=weights)

                # Loop through the results for each k and save/summarize
                for k, result in results_by_k.items():
                    print(f"--- Post-Processing results for k={k} ---")

                    # Save detailed JSON result for this run and top-k
                    config_basename, _ = os.path.splitext(os.path.basename(config_path))
                    result_filename = os.path.join(lambda_results_dir, f"{i}_{config_basename}_k{k}.json")
                    with open(result_filename, "w", encoding='utf-8') as f:
                        f.write(result.model_dump_json(indent=2))

                    # Prepare the DETAILED summary row
                    row = create_summary_row(i, config_path, strategy, result, k)
                    summary_rows.append(row)

                    print(f"  Overall Avg Recall:    {100 * result.avg_recall: .2f}%")
                    print(f"  Overall Avg Precision: {100 * result.avg_precision: .2f}%")
                    print(f"  Overall Avg F1-Score:  {100 * result.avg_f1_score: .2f}%")

            except Exception as e:
                import traceback
                print(f"!!!!!!!!!!!! ERROR running benchmark for config {config_path} !!!!!!!!!!!!")
                print(f"Error: {e}")
                traceback.print_exc()
                summary_rows.append(
                    {"config_file": config_path, "recall": "ERROR", "precision": "ERROR", "f1_score": "ERROR"})

        # Save final summary CSV and STATS for this lambda
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(lambda_results_dir, "results_summary.csv")
            df.to_csv(summary_path, index=False)
            print(f'\\nLambda {lambda_val} summary saved to: "{summary_path}"')

    stats_tracker.stop_timer('overall_run')
    stats_report_content = stats_tracker.report()
    stats_path = os.path.join(results_dir, "stats.txt")
    try:
        with open(stats_path, "w", encoding='utf-8') as f:
            f.write(stats_report_content)
    except Exception as e:
        print(f"Error saving stats report: {e}. Skipping...")

    print(f"\\nBenchmark run '{run_name}' finished.")

    end_time = datetime.now()
    print(f"Run finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LegalBench-RAG-mini MMR Evaluation")
    parser.add_argument(
        "--retrieval-configs", "-rc",
        nargs='+', required=True,
        help="One or more paths to retrieval strategy JSON config files."
    )
    parser.add_argument(
        "--max-tests-per-benchmark", "-m", type=int, default=194,
        help="Maximum number of tests to sample from each sub-benchmark (e.g., cuad, maud)."
    )
    parser.add_argument(
        "--sort-by-document", action="store_true",
        help="Enable sorting by document to potentially speed up ingestion during testing."
    )
    parser.add_argument(
        "--seed", type=int, default=None
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results/legalbenchrag_mmr",
        help="Base directory to save the output run folder."
    )

    args = parser.parse_args()
    asyncio.run(main(args))
"""

full_script = header_content + mmr_class_code + new_main_code

with open(dest_path, 'w', encoding='utf-8') as f:
    f.write(full_script)
