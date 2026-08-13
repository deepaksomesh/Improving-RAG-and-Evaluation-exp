import time
import asyncio
import os
import pathlib
from sac_rag.utils.hyde import generate_legal_hypothesis
from sac_rag.utils.ai import ai_embedding, AIEmbeddingModel, AIEmbeddingType

os.environ["SAC_CACHE_DIR"] = str(pathlib.Path.cwd() / "data" / "cache_mini")

async def test():
    query = "Consider \"Wordscapes\"'s privacy policy; does it collect my location?"
    dataset_name = "privacy_qa"
    
    print("Testing generate_legal_hypothesis...")
    t0 = time.time()
    hypothesis = await generate_legal_hypothesis(query, dataset_name)
    t1 = time.time()
    print(f"Hypothesis time: {t1-t0:.4f}s")
    
    combined_query = f"{query}\n\n{hypothesis}"
    
    model = AIEmbeddingModel(company="google", model="gemini-embedding-001")
    print("Testing ai_embedding...")
    t0 = time.time()
    emb = await ai_embedding(model, [combined_query], AIEmbeddingType.QUERY)
    t1 = time.time()
    print(f"Embedding time: {t1-t0:.4f}s")

asyncio.run(test())
