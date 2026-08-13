import asyncio
import pathlib
import os
os.environ["SAC_CACHE_DIR"] = str(pathlib.Path.cwd() / "data" / "cache_mini")
os.environ["SAC_SUMMARIES_DIR"] = str(pathlib.Path.cwd() / "data" / "summary_mini")

import sys
import argparse
from datetime import datetime

# Add both the project root and the src directory to the python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src'))

from sac_rag.data_models import Document, QAGroundTruth
from sac_rag.utils.credentials import credentials
from sac_rag.utils.retriever_factory import create_retriever
from sac_rag.methods.baseline import BaselineRetrievalStrategy
from sac_rag.utils.chunking import ChunkingStrategy
from sac_rag.utils.ai import AIEmbeddingModel

from benchmarks.legalbenchrag.run_benchmark_mini import setup_and_load_data, QAResult, BenchmarkResult
from sac_rag.methods.baseline import BaselineRetrievalMethod, EmbeddingInfo
from sac_rag.utils.chunking import get_chunks, Chunk
from sac_rag.utils.ai import ai_embedding, AIEmbeddingType, generate_document_summary, ai_rerank
from sac_rag.utils.stats_tracker import stats_tracker
from sac_rag.data_models import QueryResponse, RetrievedSnippet
from tqdm import tqdm
import numpy as np

class NumpyBaselineRetrievalMethod(BaselineRetrievalMethod):
    def __init__(self, retrieval_strategy: BaselineRetrievalStrategy, cache_dir=None):
        super().__init__(retrieval_strategy, cache_dir)
        self.embeddings_matrix = None

    async def sync_all_documents(self) -> None:
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"NumpyBaseline: Calculating chunks using strategy '{self.retrieval_strategy.chunking_strategy.strategy_name}'...")

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

        all_chunks = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        stats_tracker.set('chunks_created', len(all_chunks))
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"NumpyBaseline: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            self.embedding_infos = []
            return

        self.embedding_infos = []

        stats_tracker.start_timer('embedding_generation')
        progress_bar = tqdm(total=len(all_chunks), desc="NumpyBaseline: Processing Embeddings", ncols=100)
        def progress_callback():
            if progress_bar:
                progress_bar.update(1)

        # Prepend the document name
        chunk_contents = [
            f"Document Name: {os.path.basename(chunk.file_path)}\n\n{chunk.content}" 
            for chunk in all_chunks
        ]
        embeddings = await ai_embedding(
            self.retrieval_strategy.embedding_model,
            chunk_contents,
            AIEmbeddingType.DOCUMENT,
            callback=progress_callback,
        )
        if progress_bar:
            progress_bar.close()
        stats_tracker.stop_timer('embedding_generation')

        print(f"NumpyBaseline: Start indexing embeddings using Numpy...")
        self.embeddings_matrix = np.array(embeddings, dtype=np.float32)
        
        for chunk in all_chunks:
            self.embedding_infos.append(
                EmbeddingInfo(
                    document_id=chunk.file_path,
                    span=chunk.span,
                    processed_content=chunk.content,
                )
            )
        print(f"NumpyBaseline: Finished indexing {len(self.embedding_infos)} embeddings.")

    async def query(self, query: str) -> QueryResponse:
        if self.embeddings_matrix is None or not self.embedding_infos:
            return QueryResponse(retrieved_snippets=[])

        query_embedding = (await ai_embedding(
            self.retrieval_strategy.embedding_model, [query], AIEmbeddingType.QUERY
        ))[0]

        query_emb = np.array(query_embedding, dtype=np.float32)
        
        # Calculate exact L2 distance
        distances = np.linalg.norm(self.embeddings_matrix - query_emb, axis=1)
        
        # Get top K
        top_k = self.retrieval_strategy.embedding_top_k
        top_k_indices = np.argsort(distances)[:top_k]

        retrieved_metadatas = [self.embedding_infos[idx] for idx in top_k_indices]

        # 3. Get the top retrieval snippets, up until the token limit
        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets = []
        for i, metadata in enumerate(retrieved_metadatas):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            span = metadata.span
            text_content = self.get_embedding_info_text(metadata)
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


async def main(args):
    # Set necessary API Keys for embeddings
    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()

    print(f"Loading LegalBench-RAG data... (max tests: {args.max_tests})")
    # Use the native data loader to ensure we load the exact same way as the standard benchmarks
    corpus, tests, weights = setup_and_load_data(max_tests=args.max_tests, sort_by_doc=False, seed=42)

    print(f"Loaded {len(corpus)} documents and {len(tests)} queries.")

    # 1. Define Standard RAG Strategy (RCTS, NO SAC)
    strategy = BaselineRetrievalStrategy(
        strategy_type="baseline",
        chunking_strategy=ChunkingStrategy(
            strategy_name="rcts",
            chunk_size=2048,  # ~512 tokens
            chunk_overlap_ratio=0.1
        ),
        embedding_model=AIEmbeddingModel(
            company="google",
            model="gemini-embedding-001"
        ),
        embedding_top_k=64,  # Increased to 64 to evaluate up to K=64
        rerank_model=None,
        rerank_top_k=[64],
        token_limit=None
    )

    # 2. Create Retriever & Ingest Corpus
    print("Initializing retriever and chunking/embedding documents...")
    retriever = NumpyBaselineRetrievalMethod(strategy)
    for document in corpus:
        await retriever.ingest_document(document)
    await retriever.sync_all_documents()
    print("Ingestion complete.")

    # 3. Query Execution
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    from collections import defaultdict
    results_by_dataset = defaultdict(list)
    
    for i, test in enumerate(tests):
        print(f"Processing query {i+1}/{len(tests)}...")
        query_response = await retriever.query(test.query)
        dataset_name = test.tags[0] if test.tags else "unknown"
        results_by_dataset[dataset_name].append((test, query_response.retrieved_snippets))
        
    await retriever.cleanup()

    # 4. Compute Metrics and Write Logs Per Dataset
    k_levels = [1, 3, 5, 10, 16, 32, 64]
    
    for dataset_name, queries_results in results_by_dataset.items():
        print(f"\n=====================================")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"=====================================")
        
        # Write Debug Log for this dataset
        log_path = os.path.join(os.path.dirname(args.log_file), f"debug_log_docname_{dataset_name}.txt")
        with open(log_path, 'w', encoding='utf-8') as log_f:
            log_f.write(f"=== Debug RAG Evaluation Run ===\n")
            log_f.write(f"Dataset: {dataset_name}\n")
            log_f.write(f"Method: DocName-SAC (Prepend)\n")
            log_f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"================================\n\n")

            for i, (test, snippets) in enumerate(queries_results):
                # We only log top 10 to keep the file readable
                top_10_snippets = snippets[:10]
                
                log_f.write(f"--------------------------------------------------\n")
                log_f.write(f"QUERY [{i+1}/{len(queries_results)}]:\n")
                log_f.write(f"{test.query}\n\n")
                
                log_f.write(f"--- GOLD ANSWER (GROUND TRUTH SNIPPETS) ---\n")
                for j, gt in enumerate(test.snippets):
                    log_f.write(f"  [GT {j+1}] Document: {gt.file_path}\n")
                    log_f.write(f"  [GT {j+1}] Span: {gt.span}\n")
                log_f.write("\n")
                
                log_f.write(f"--- TOP-10 RETRIEVED CHUNKS ---\n")
                for j, snippet in enumerate(top_10_snippets):
                    log_f.write(f"\n  >>> RANK {j+1} <<<\n")
                    log_f.write(f"  Document Name : {snippet.file_path}\n")
                    log_f.write(f"  Chunk Span    : {snippet.span}\n")
                    log_f.write(f"  Chunk Content :\n")
                    log_f.write(f"\"\"\"\n{snippet.full_chunk_text}\n\"\"\"\n")
                
                log_f.write(f"\n--------------------------------------------------\n\n")
        
        print(f"Detailed logs saved to: {log_path}")

        # Compute Metrics at different K levels
        print(f"\nMetrics at different K levels:")
        for k in k_levels:
            qa_results_for_k = []
            drm_scores = []
            
            for test, snippets in queries_results:
                sliced_snippets = snippets[:k]
                qa_res = QAResult(qa_gt=test, retrieved_snippets=sliced_snippets)
                qa_results_for_k.append(qa_res)
                
                # Calculate DRM for this query
                ground_truth_docs = {gt.file_path for gt in test.snippets}
                incorrect_retrievals = sum(1 for s in sliced_snippets if s.file_path not in ground_truth_docs)
                drm = incorrect_retrievals / len(sliced_snippets) if sliced_snippets else 0.0
                drm_scores.append(drm)
            
            # Unweighted average for this dataset subset
            avg_recall = sum(r.recall for r in qa_results_for_k) / len(qa_results_for_k)
            avg_precision = sum(r.precision for r in qa_results_for_k) / len(qa_results_for_k)
            avg_drm = sum(drm_scores) / len(drm_scores) if drm_scores else 0.0
            
            avg_f1 = 0.0
            if avg_precision + avg_recall > 0:
                avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                
            print(f"  K={k:<2} | DRM: {avg_drm * 100:>5.2f}% | Recall: {avg_recall * 100:>5.2f}% | Precision: {avg_precision * 100:>5.2f}% | F1: {avg_f1 * 100:>5.2f}%")
class Logger(object):
    def __init__(self, filename="debug_evaluation/debug_log_docname.txt", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaulting to -1 to load all queries in the mini dataset
    parser.add_argument("--max-tests", type=int, default=-1, help="Number of queries to run")
    parser.add_argument("--log-file", type=str, default="debug_evaluation/debug_log_docname.txt")
    args = parser.parse_args()
    
    # Create or clear the log file
    open(args.log_file, "w", encoding="utf-8").close()
    
    sys.stdout = Logger(args.log_file, sys.stdout)
    sys.stderr = Logger(args.log_file, sys.stderr)

    asyncio.run(main(args))
