import re

with open("benchmarks/legalbenchrag/run_benchmark_mini.py", "r", encoding="utf-8") as f:
    content = f.read()

# The code we want to inject
numpy_class_code = """
import numpy as np
from sac_rag.methods.baseline import BaselineRetrievalMethod, EmbeddingInfo
from sac_rag.utils.chunking import get_chunks, Chunk
from sac_rag.utils.ai import ai_embedding, AIEmbeddingType
from sac_rag.utils.stats_tracker import stats_tracker
from sac_rag.data_models import QueryResponse, RetrievedSnippet
from tqdm import tqdm
import asyncio

class NumpyBaselineRetrievalMethod(BaselineRetrievalMethod):
    def __init__(self, retrieval_strategy, cache_dir=None):
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

        chunk_contents = [chunk.content for chunk in all_chunks]
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
        
        distances = np.linalg.norm(self.embeddings_matrix - query_emb, axis=1)
        
        top_k = self.retrieval_strategy.embedding_top_k
        top_k_indices = np.argsort(distances)[:top_k]

        retrieved_metadatas = [self.embedding_infos[idx] for idx in top_k_indices]

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

import sac_rag.utils.retriever_factory
sac_rag.utils.retriever_factory.BaselineRetrievalMethod = NumpyBaselineRetrievalMethod
"""

# Inject it after the imports
content = content.replace("from sac_rag.utils.utils import sanitize_filename\n", "from sac_rag.utils.utils import sanitize_filename\n\n" + numpy_class_code + "\n")

with open("benchmarks/legalbenchrag/run_benchmark_mini.py", "w", encoding="utf-8") as f:
    f.write(content)
