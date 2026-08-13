import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_drm_file(filepath):
    """Parses the k4_filepath_analysis.txt and returns DRM for each dataset."""
    drm_data = {}
    if not os.path.exists(filepath):
        return drm_data
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Match Dataset: <name> and then Correct File Paths: <num> / <num> (<percent>%)
    blocks = content.split("Dataset: ")
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        dataset_name = lines[0].strip().lower()
        
        for line in lines[1:]:
            if "Incorrect File Paths:" in line:
                # Extract percentage
                match = re.search(r"\(([\d.]+)%\)", line)
                if match:
                    drm_percent = float(match.group(1))
                    drm_data[dataset_name] = drm_percent
                    
    return drm_data

def generate_centroid_plots():
    base_dir = Path("results/legalbenchrag_centroid")
    
    # Find all alpha folders
    alpha_folders = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.startswith("alpha_"):
                alpha_val = float(d.split("_")[1])
                alpha_folders.append((alpha_val, os.path.join(root, d)))
                
    alpha_folders.sort(key=lambda x: x[0])  # Sort by alpha value
    alphas_found = [a[0] for a in alpha_folders]
    
    datasets = ['contractnli', 'cuad', 'maud', 'privacy_qa']
    
    # Data structure for DRM
    drm_data = {dataset: [] for dataset in datasets}
    
    # Data structure for Recall/Precision at all Ks
    # recall_data[alpha][k] = val
    recall_data = {alpha: {} for alpha in alphas_found}
    precision_data = {alpha: {} for alpha in alphas_found}
    
    all_ks = set()
    
    for alpha_val, folder_path in alpha_folders:
        # 1. Parse CSV for Recall/Precision at all Ks (overall average, not subset specific)
        csv_path = os.path.join(folder_path, "results_summary.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # The overall recall is 'recall', precision is 'precision', k is 'rerank_top_k'
            for _, row in df.iterrows():
                k = int(row['rerank_top_k'])
                all_ks.add(k)
                recall_data[alpha_val][k] = row['recall'] * 100
                precision_data[alpha_val][k] = row['precision'] * 100
        
        # 2. Parse DRM file for k=4
        drm_files = glob.glob(os.path.join(folder_path, "*k4_filepath_analysis.txt"))
        if drm_files:
            drm_path = drm_files[0] # Take the first one
            drm_parsed = parse_drm_file(drm_path)
            for dataset in datasets:
                if dataset in drm_parsed:
                    drm_data[dataset].append(drm_parsed[dataset])
                else:
                    drm_data[dataset].append(0)
        else:
            for dataset in datasets:
                drm_data[dataset].append(0)

    all_ks_sorted = sorted(list(all_ks))

    out_dir = Path("debug_evaluation/centroid_plots_custom")
    os.makedirs(out_dir, exist_ok=True)
    
    # PLOT 1: DRM vs Alpha (all datasets on one plot)
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'd']
    for i, dataset in enumerate(datasets):
        plt.plot(alphas_found, drm_data[dataset], marker=markers[i], linewidth=2, markersize=8, label=dataset.upper())
    
    plt.title('Incorrect File Paths (%) at K=4 vs. Alpha', fontsize=16, fontweight='bold')
    plt.xlabel('Alpha (Centroid Interpolation Weight)', fontsize=14)
    plt.ylabel('Incorrect File Paths (%)', fontsize=14)
    plt.xticks(alphas_found)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    drm_output = out_dir / "centroid_incorrect_filepaths_all_datasets.png"
    plt.savefig(drm_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved DRM plot to {drm_output}")

    # PLOT 2: Overall Average Recall vs K (lines are alphas)
    plt.figure(figsize=(10, 6))
    for i, alpha in enumerate(alphas_found):
        y_vals = [recall_data[alpha].get(k, 0) for k in all_ks_sorted]
        plt.plot(all_ks_sorted, y_vals, marker=markers[i%len(markers)], linewidth=2, markersize=8, label=f"Alpha={alpha}")
        
    plt.title('Overall Average Recall vs. K', fontsize=16, fontweight='bold')
    plt.xlabel('K (Number of chunks retrieved)', fontsize=14)
    plt.ylabel('Average Recall (%)', fontsize=14)
    plt.xticks(all_ks_sorted)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    recall_output = out_dir / "centroid_overall_recall.png"
    plt.savefig(recall_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Recall plot to {recall_output}")

    # PLOT 3: Overall Average Precision vs K (lines are alphas)
    plt.figure(figsize=(10, 6))
    for i, alpha in enumerate(alphas_found):
        y_vals = [precision_data[alpha].get(k, 0) for k in all_ks_sorted]
        plt.plot(all_ks_sorted, y_vals, marker=markers[i%len(markers)], linewidth=2, markersize=8, label=f"Alpha={alpha}")
        
    plt.title('Overall Average Precision vs. K', fontsize=16, fontweight='bold')
    plt.xlabel('K (Number of chunks retrieved)', fontsize=14)
    plt.ylabel('Average Precision (%)', fontsize=14)
    plt.xticks(all_ks_sorted)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    precision_output = out_dir / "centroid_overall_precision.png"
    plt.savefig(precision_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision plot to {precision_output}")

if __name__ == "__main__":
    generate_centroid_plots()
