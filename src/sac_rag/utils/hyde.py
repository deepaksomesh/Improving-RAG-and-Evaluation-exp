import os
import pathlib
import hashlib
import asyncio
import logging
from typing import Optional

from sac_rag.utils.ai import get_ai_connection, cache
from sac_rag.utils.credentials import credentials

logger = logging.getLogger(__name__)

async def generate_legal_hypothesis(query: str, dataset_name: str) -> str:
    """
    Generates a formal legal hypothesis for a given query, tailored to the dataset.
    Caches the result in diskcache and as a .txt file for manual inspection.
    """
    ai_conn = await get_ai_connection()
    if ai_conn.openrouter_client is None:
        raise ValueError("OpenRouter client is not initialized. Please set openrouter_api_key in credentials.")

    dataset_context = {
        "contractnli": "Non-Disclosure Agreement (NDA). Focus on confidentiality, intellectual property, and sharing restrictions.",
        "maud": "Merger Agreement. Focus on corporate acquisitions, material adverse effects, closing conditions, and shareholder rights.",
        "cuad": "Commercial Contract. Focus on commercial terms, payment, termination, governing law, and indemnification.",
        "privacy_qa": "Privacy Policy. Focus on user data collection, GDPR/CCPA compliance, cookies, and data sharing."
    }

    context = dataset_context.get(dataset_name, "formal legal contract")

    prompt = f"""SYSTEM:
You are an expert corporate lawyer. The user will ask a question regarding a specific type of legal document: a {context}.
Your task is to generate the EXACT, formal legal boilerplate clause that would contain the answer to the user's question. 
Do not answer the question. Do not explain anything. 
Write a highly realistic, abstract legal clause using formal jargon that perfectly aligns with the semantic structure of a {context}.

USER: 
{query}

HYPOTHETICAL CLAUSE:
"""

    # Create a unique cache key based on the query and dataset
    hash_input = f"hyde_v1|||{dataset_name}|||{query}"
    cache_key = hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    # 1. Check diskcache
    cached_hypothesis = cache.get(cache_key)
    if cached_hypothesis is not None:
        return cached_hypothesis

    # 2. Check local txt file cache
    cache_dir_base = os.environ.get("SAC_CACHE_DIR", str(pathlib.Path.cwd() / "data" / "cache"))
    hyde_cache_dir = pathlib.Path(cache_dir_base) / "hyde_mini" / dataset_name
    hyde_cache_dir.mkdir(parents=True, exist_ok=True)
    
    txt_cache_file = hyde_cache_dir / f"{cache_key}.txt"
    if txt_cache_file.exists():
        with open(txt_cache_file, 'r', encoding='utf-8') as f:
            hypothesis = f.read().strip()
            if hypothesis:
                cache.set(cache_key, hypothesis)
                return hypothesis

    # 3. Generate using OpenRouter (gpt-4o-mini)
    logger.info(f"Generating Legal-HyDE for query: '{query[:30]}...' (Dataset: {dataset_name})")
    
    # We use a small retry loop to handle transient API issues
    for attempt in range(3):
        try:
            response = await ai_conn.openrouter_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            hypothesis = response.choices[0].message.content.strip()
            
            # Save to diskcache
            cache.set(cache_key, hypothesis)
            
            # Save to txt file
            with open(txt_cache_file, 'w', encoding='utf-8') as f:
                f.write(hypothesis)
                
            return hypothesis
        except Exception as e:
            logger.warning(f"HyDE Generation failed (attempt {attempt+1}/3): {e}")
            await asyncio.sleep(2 ** attempt)
            
    logger.error("Failed to generate HyDE after 3 attempts. Returning original query.")
    return query
