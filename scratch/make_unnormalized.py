import sys

patch = """import sac_rag.utils.ai
original_ai_embedding = sac_rag.utils.ai.ai_embedding

async def unnormalized_ai_embedding(*args, **kwargs):
    kwargs['normalize'] = False
    return await original_ai_embedding(*args, **kwargs)

sac_rag.utils.ai.ai_embedding = unnormalized_ai_embedding
import sac_rag.methods.baseline
sac_rag.methods.baseline.ai_embedding = unnormalized_ai_embedding

"""

with open('benchmarks/legalbenchrag/run_benchmark_mini_docname.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure results go to a new folder so they don't overwrite
content = content.replace('results/legalbenchrag', 'results/legalbenchrag_docname_unnormalized')

with open('benchmarks/legalbenchrag/run_benchmark_docname_unnormalized.py', 'w', encoding='utf-8') as f:
    f.write(patch + content)

print("Created run_benchmark_docname_unnormalized.py")
