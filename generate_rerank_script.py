import os

in_path = 'd:/Thesis/Baselines/summary-augmented-chunking/benchmarks/legalbenchrag/run_benchmark_mini_clustered_centroid.py'
out_path = 'd:/Thesis/Baselines/summary-augmented-chunking/benchmarks/legalbenchrag/run_benchmark_mini_clustered_centroid_rerank.py'

with open(in_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Replace the output dir so it saves in a distinct place
code = code.replace('legalbenchrag_clustered_centroid', 'legalbenchrag_clustered_centroid_rerank')

# Inject the rerank configuration override
injection_target = "strategy = load_strategy_from_file(config_path)"
injection_code = """strategy = load_strategy_from_file(config_path)
                
                # FORCE COHERE RERANKING
                from sac_rag.utils.ai import AIRerankModel
                print("Forcing Cohere Rerank injection into strategy.")
                strategy.rerank_model = AIRerankModel(company="cohere", model="rerank-english-v3.0")
                if not strategy.rerank_top_k:
                    strategy.rerank_top_k = [1, 2, 4]"""

code = code.replace(injection_target, injection_code)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(code)
