import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(filepath):
    """Parses a debug_log_*.txt file and extracts metrics for each dataset and each K."""
    metrics_by_dataset = {}
    if not os.path.exists(filepath):
        return metrics_by_dataset
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split content by dataset blocks
    dataset_blocks = re.split(r"={30,}\nDataset:\s*([A-Z_]+)\n={30,}", content)
    
    # dataset_blocks[0] is preamble. 
    # dataset_blocks[1] is FIRST DATASET NAME, dataset_blocks[2] is its content
    # dataset_blocks[3] is SECOND DATASET NAME, dataset_blocks[4] is its content
    for i in range(1, len(dataset_blocks), 2):
        dataset_name = dataset_blocks[i].strip().lower()
        block_content = dataset_blocks[i+1]
        
        metrics_by_k = {}
        pattern = r"K=(\d+)\s*\|\s*DRM:\s*([\d.]+)%\s*\|\s*Recall:\s*([\d.]+)%\s*\|\s*Precision:\s*([\d.]+)%\s*\|\s*F1:\s*([\d.]+)%"
        for match in re.finditer(pattern, block_content):
            k = int(match.group(1))
            drm = float(match.group(2))
            recall = float(match.group(3))
            precision = float(match.group(4))
            f1 = float(match.group(5))
            
            metrics_by_k[k] = {
                'DRM': drm,
                'Recall': recall,
                'Precision': precision,
                'F1': f1
            }
        metrics_by_dataset[dataset_name] = metrics_by_k
        
    return metrics_by_dataset

def generate_comparative_plots():
    datasets = ['contractnli', 'cuad', 'maud', 'privacy_qa']
    methods = {
        'Standard': 'debug_log.txt',
        'DocName': 'debug_log_docname.txt',
        'SAC': 'debug_log_sac.txt'
    }
    
    metrics_to_plot = ['Recall', 'Precision', 'DRM', 'F1']
    colors = {'Standard': '#1f77b4', 'DocName': '#ff7f0e', 'SAC': '#2ca02c'}
    markers = {'Standard': 'o', 'DocName': 's', 'SAC': '^'}
    
    base_dir = Path("debug_evaluation")
    
    # Load all data first
    all_data = {}
    for method_name, filename in methods.items():
        filepath = base_dir / filename
        all_data[method_name] = parse_log_file(filepath)
    
    for dataset in datasets:
        print(f"Generating plots for {dataset.upper()}...")
        dataset_data = {}
        
        for method_name in methods.keys():
            if dataset in all_data[method_name]:
                dataset_data[method_name] = all_data[method_name][dataset]
        
        if not dataset_data:
            print(f"  No data found for {dataset}, skipping.")
            continue
            
        # We need all methods to have data to compare cleanly, but we'll plot whatever is available
        available_ks = sorted(list(set(k for method_data in dataset_data.values() for k in method_data.keys())))
        
        # Create a figure with 2x2 subplots (for the 4 metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'RAG Method Comparison on {dataset.upper()}', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics_to_plot):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            for method_name in methods.keys():
                if method_name not in dataset_data:
                    continue
                    
                method_metrics = dataset_data[method_name]
                x_vals = [k for k in available_ks if k in method_metrics]
                y_vals = [method_metrics[k][metric] for k in x_vals]
                
                ax.plot(x_vals, y_vals, marker=markers[method_name], color=colors[method_name], 
                        linewidth=2, markersize=8, label=method_name)
            
            ax.set_title(f'{metric} vs. K', fontsize=14)
            ax.set_xlabel('K (Number of chunks retrieved)', fontsize=12)
            ax.set_ylabel(f'{metric} (%)', fontsize=12)
            ax.set_xticks(available_ks)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=11)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_file = base_dir / f"comparative_plot_{dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to {output_file}")

if __name__ == "__main__":
    generate_comparative_plots()
