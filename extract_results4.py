import csv
import re

configs = [
    ("SAC", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag\2026-06-19_11-09-30\results_summary.csv", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag\2026-06-19_11-09-30\0_default_k4_filepath_analysis.txt"),
    ("Centroid", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_centroid\2026-06-19_15-54-56\alpha_0.25\results_summary.csv", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_centroid\2026-06-19_15-54-56\alpha_0.25\0_standard_rag_k4_filepath_analysis.txt"),
    ("Clustered Centroid", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid\2026-06-29_15-15-01\alpha_0.25\results_summary.csv", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid\2026-06-29_15-15-01\alpha_0.25\0_default_k4_filepath_analysis.txt"),
    ("Clustered Centroid Docname", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_docname\2026-06-29_15-24-31\alpha_0.25\results_summary.csv", 
     None),
    ("Clustered Centroid Hybrid", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_hybrid\2026-06-28_17-19-34\alpha_0.25\results_summary.csv", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_hybrid\2026-06-28_17-19-34\alpha_0.25\0_default_k4_filepath_analysis.txt"),
    ("Clustered Centroid Legal HyDE", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_legal_hyde\2026-06-29_17-09-15\alpha_0.25\results_summary.csv", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_legal_hyde\2026-06-29_17-09-15\alpha_0.25\0_default_k4_filepath_analysis.txt"),
    ("Clustered Centroid Pure", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_pure\2026-06-29_16-03-03\alpha_0.25\results_summary.csv", 
     None),
    ("Clustered Centroid Rerank", 
     r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag_clustered_centroid_rerank\2026-06-28_19-15-48\alpha_0.25\results_summary.csv", 
     None)
]

print(f"{'Methodology':<35} | {'K=1':<6} | {'K=2':<6} | {'K=4':<6} | {'K=8':<6} | {'DRM@4':<6}")
print("-" * 80)

for name, csv_path, txt_path in configs:
    k1, k2, k4, k8, drm = "N/A", "N/A", "N/A", "N/A", "N/A"
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row.get('rerank_top_k', row.get('k', '0'))
                if not k: continue
                k = int(k)
                recall = float(row.get('recall', 0))
                val = f"{recall*100:.1f}%"
                if k == 1: k1 = val
                if k == 2: k2 = val
                if k == 4: k4 = val
                if k == 8: k8 = val
    except Exception:
        pass
        
    if txt_path:
        try:
            with open(txt_path, 'r') as f:
                content = f.read()
                matches = re.findall(r'Incorrect File Paths:\s+(\d+)\s+/\s+(\d+)', content)
                if matches:
                    total_incorrect = sum(int(m[0]) for m in matches)
                    total_queries = sum(int(m[1]) for m in matches)
                    if total_queries > 0:
                        drm = f"{(total_incorrect / total_queries) * 100:.1f}%"
        except Exception:
            pass
            
    print(f"{name:<35} | {k1:<6} | {k2:<6} | {k4:<6} | {k8:<6} | {drm:<6}")

