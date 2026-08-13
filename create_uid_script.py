import os
import re

source_path = 'benchmarks/legalbenchrag/run_benchmark_mini.py'
dest_path = 'benchmarks/legalbenchrag/run_benchmark_mini_uid.py'

with open(source_path, 'r', encoding='utf-8') as f:
    content = f.read()

# We need to inject the UID patch
uid_patch = """
import os
from sac_rag.methods.baseline import BaselineRetrievalMethod
from sac_rag.utils.chunking import get_chunks, Chunk
from sac_rag.utils.ai import ai_embedding, AIEmbeddingType
from sac_rag.utils.stats_tracker import stats_tracker
from tqdm import tqdm
import asyncio
import sqlite3
import sqlite_vec
from typing import List

class UIDBaselineRetrievalMethod(BaselineRetrievalMethod):
    async def sync_all_documents(self) -> None:
        # 1. Calculate chunks using the shared utility
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"UIDBaseline: Calculating chunks using strategy '{self.retrieval_strategy.chunking_strategy.strategy_name}'...")

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
        print(f"UIDBaseline: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            self.embedding_infos = []
            return

        self.embedding_infos = []
        stats_tracker.start_timer('embedding_generation')
        progress_bar = tqdm(total=len(all_chunks), desc="UIDBaseline: Processing Embeddings", ncols=100)

        def progress_callback():
            if progress_bar:
                progress_bar.update(1)

        # ====== THE UID INJECTION ======
        doc_id_map = {}
        next_id = 1
        for chunk in all_chunks:
            if chunk.file_path not in doc_id_map:
                doc_id_map[chunk.file_path] = f"DOC_{next_id:04d}"
                next_id += 1

        chunk_contents = [
            f"Document Identifier: {doc_id_map[chunk.file_path]}\\n\\n{chunk.content}" 
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

        print(f"UIDBaseline: Start indexing embeddings into SQLite...")

        if self.sqlite_db is None:
            if os.path.exists(self.sqlite_db_file_path):
                os.remove(self.sqlite_db_file_path)
            self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
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
                        processed_content=chunk.content,
                    )
                )

        print(f"UIDBaseline: Finished indexing {len(self.embedding_infos)} embeddings.")

import sac_rag.utils.retriever_factory
from sac_rag.methods.baseline import serialize_f32, EmbeddingInfo
sac_rag.utils.retriever_factory.BaselineRetrievalMethod = UIDBaselineRetrievalMethod
"""

content = content.replace("from sac_rag.utils.utils import sanitize_filename\n", "from sac_rag.utils.utils import sanitize_filename\n\n" + uid_patch + "\n")

# Make sure it logs its results to a different folder so they don't get mixed up
content = content.replace('results/legalbenchrag', 'results/legalbenchrag_uid')

with open(dest_path, 'w', encoding='utf-8') as f:
    f.write(content)
