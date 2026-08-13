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
            if "Correct File Paths:" in line:
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
    
    datasets = ['contractnli', 'cuad', 'maud', 'privacy_qa']
    metrics = ['Recall', 'Precision', 'DRM']
    
    # Store data
    plot_data = {dataset: {m: [] for m in metrics} for dataset in datasets}
    alphas_found = []
    
    for alpha_val, folder_path in alpha_folders:
        alphas_found.append(alpha_val)
        
        # 1. Parse CSV for Recall/Precision at k=4
        csv_path = os.path.join(folder_path, "results_summary.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter for K=4 (stored in rerank_top_k column in these old benchmarks)
            k4_row = df[df['rerank_top_k'] == 4]
            
            if not k4_row.empty:
                for dataset in datasets:
                    recall_col = f"{dataset}|recall"
                    precision_col = f"{dataset}|precision"
                    
                    if recall_col in df.columns:
                        # Convert to percentage
                        plot_data[dataset]['Recall'].append(k4_row.iloc[0][recall_col] * 100)
                        plot_data[dataset]['Precision'].append(k4_row.iloc[0][precision_col] * 100)
                    else:
                        plot_data[dataset]['Recall'].append(0)
                        plot_data[dataset]['Precision'].append(0)
            else:
                for dataset in datasets:
                    plot_data[dataset]['Recall'].append(0)
                    plot_data[dataset]['Precision'].append(0)
        
        # 2. Parse DRM file
        drm_files = glob.glob(os.path.join(folder_path, "*k4_filepath_analysis.txt"))
        if drm_files:
            drm_path = drm_files[0] # Take the first one
            drm_parsed = parse_drm_file(drm_path)
            for dataset in datasets:
                if dataset in drm_parsed:
                    plot_data[dataset]['DRM'].append(drm_parsed[dataset])
                else:
                    plot_data[dataset]['DRM'].append(0)
        else:
            for dataset in datasets:
                plot_data[dataset]['DRM'].append(0)

    # Now generate the plots!
    out_dir = Path("debug_evaluation/centroid_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    colors = {'Recall': '#1f77b4', 'Precision': '#ff7f0e', 'DRM': '#2ca02c'}
    markers = {'Recall': 'o', 'Precision': 's', 'DRM': '^'}
    
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        
        for metric in metrics:
            y_vals = plot_data[dataset][metric]
            plt.plot(alphas_found, y_vals, marker=markers[metric], color=colors[metric], 
                     linewidth=2, markersize=8, label=metric)
                     
        plt.title(f'Centroid Anchoring: {dataset.upper()} at K=4', fontsize=16, fontweight='bold')
        plt.xlabel('Alpha (Centroid Interpolation Weight)', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(alphas_found)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        output_file = out_dir / f"centroid_k4_{dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    generate_centroid_plots()
