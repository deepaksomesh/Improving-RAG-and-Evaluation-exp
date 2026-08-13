import asyncio
import os
import sys

sys.path.append(os.path.abspath('src'))

from sac_rag.utils.ai import ai_embedding, AIEmbeddingType, AIModel

async def main():
    model = AIModel(company='openrouter', model='google/gemini-embedding-001')
    try:
        res = await ai_embedding(model, [''], AIEmbeddingType.QUERY)
        print('Empty string success:', len(res))
    except Exception as e:
        print('Empty string failed:', repr(e))

    try:
        res = await ai_embedding(model, ['What is the termination clause?'], AIEmbeddingType.QUERY)
        print('Normal string success:', len(res))
    except Exception as e:
        print('Normal string failed:', repr(e))

asyncio.run(main())
