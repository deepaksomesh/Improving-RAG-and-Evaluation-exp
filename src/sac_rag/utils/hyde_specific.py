import os
import pathlib
import hashlib
import asyncio
import logging
from typing import Optional

from sac_rag.utils.ai import get_ai_connection, cache
from sac_rag.utils.credentials import credentials

logger = logging.getLogger(__name__)

async def generate_specific_legal_hypothesis(query: str, dataset_name: str) -> str:
    """
    Generates a highly concise, specific search phrase for a given query, tailored to the dataset.
    Prevents hallucination of specific entities, dates, or numbers.
    Caches the result in diskcache and as a .txt file for manual inspection.
    """
    ai_conn = await get_ai_connection()
    if ai_conn.openrouter_client is None:
        raise ValueError("OpenRouter client is not initialized. Please set openrouter_api_key in credentials.")

    dataset_context = {
        "contractnli": "Non-Disclosure Agreement (NDA)",
        "maud": "Merger Agreement",
        "cuad": "Commercial Contract",
        "privacy_qa": "Privacy Policy"
    }

    context = dataset_context.get(dataset_name, "formal legal contract")

    prompt = f"""SYSTEM:
You are an expert legal search assistant. The user is querying a {context}.
Extract the core legal concepts and specific intent from the user's query and generate a highly concise, specific search phrase (maximum 5-10 words).
Do NOT write a full clause. Do NOT include boilerplate. Use exact legal terminology. Do NOT leak or hallucinate specific names, dates, or numbers unless they are explicitly in the query.

Example User: What is the notice period for termination?
Example Output: notice period prior written termination

USER: 
{query}

CONCISE SEARCH PHRASE:
"""

    # Create a unique cache key based on the query and dataset
    hash_input = f"hyde_specific_v1|||{dataset_name}|||{query}"
    cache_key = hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    # 1. Check diskcache
    cached_hypothesis = cache.get(cache_key)
    if cached_hypothesis is not None:
        return cached_hypothesis

    # 2. Check local txt file cache
    cache_dir_base = os.environ.get("SAC_CACHE_DIR", str(pathlib.Path.cwd() / "data" / "cache"))
    hyde_cache_dir = pathlib.Path(cache_dir_base) / "hyde_specific" / dataset_name
    hyde_cache_dir.mkdir(parents=True, exist_ok=True)
    
    txt_cache_file = hyde_cache_dir / f"{cache_key}.txt"
    if txt_cache_file.exists():
        with open(txt_cache_file, 'r', encoding='utf-8') as f:
            hypothesis = f.read().strip()
            if hypothesis:
                cache.set(cache_key, hypothesis)
                return hypothesis

    # 3. Generate using OpenRouter (gpt-4o-mini)
    logger.info(f"Generating Specific Legal-HyDE for query: '{query[:30]}...' (Dataset: {dataset_name})")
    
    for attempt in range(3):
        try:
            response = await ai_conn.openrouter_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,  # Highly restricted
                temperature=0.1   # Low temperature for strict adherence
            )
            hypothesis = response.choices[0].message.content.strip()
            
            # Save to diskcache
            cache.set(cache_key, hypothesis)
            
            # Save to txt file
            with open(txt_cache_file, 'w', encoding='utf-8') as f:
                f.write(hypothesis)
                
            return hypothesis
        except Exception as e:
            logger.warning(f"Specific HyDE Generation failed (attempt {attempt+1}/3): {e}")
            await asyncio.sleep(2 ** attempt)
            
    logger.error("Failed to generate Specific HyDE after 3 attempts. Returning original query.")
    return query
