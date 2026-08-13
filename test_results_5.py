import json
import math
import os

results_dir = 'results/legalbenchrag_two_stage/2026-06-30_08-16-02/summary_top_n_5'
if not os.path.exists(results_dir):
    print("No summary_top_n_5 directory yet")
    exit(0)

files = [f for f in os.listdir(results_dir) if f.endswith('.json') and 'k1.' in f]

for f in files:
    with open(os.path.join(results_dir, f), 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    qa_results = data['qa_result_list']
    recalls = [r['recall'] for r in qa_results]
    precisions = [r['precision'] for r in qa_results]
    
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    f1 = 0.0
    if avg_recall + avg_precision > 0:
        f1 = 2 * (avg_recall * avg_precision) / (avg_recall + avg_precision)
        
    print(f"Top_n_5 {f} - Avg Recall: {avg_recall*100:.2f}% | Avg Precision: {avg_precision*100:.2f}% | Avg F1-Score: {f1 * 100:.2f}%")
