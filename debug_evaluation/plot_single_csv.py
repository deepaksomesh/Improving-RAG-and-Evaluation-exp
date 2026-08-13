import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_csv():
    csv_path = r"d:\Thesis\Baselines\summary-augmented-chunking\results\legalbenchrag\2026-06-19_11-09-30\results_summary.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Extract K, Recall, and Precision
    ks = df['rerank_top_k'].tolist()
    recalls = (df['recall'] * 100).tolist()
    precisions = (df['precision'] * 100).tolist()
    
    out_dir = Path("debug_evaluation/single_run_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot Recall
    plt.figure(figsize=(10, 6))
    plt.plot(ks, recalls, marker='o', linewidth=2, markersize=8, color='#1f77b4', label='Recall')
    plt.title('Overall Average Recall vs. K', fontsize=16, fontweight='bold')
    plt.xlabel('K (Number of chunks retrieved)', fontsize=14)
    plt.ylabel('Recall (%)', fontsize=14)
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    recall_out = out_dir / "single_run_recall.png"
    plt.savefig(recall_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {recall_out}")

    # Plot Precision
    plt.figure(figsize=(10, 6))
    plt.plot(ks, precisions, marker='s', linewidth=2, markersize=8, color='#ff7f0e', label='Precision')
    plt.title('Overall Average Precision vs. K', fontsize=16, fontweight='bold')
    plt.xlabel('K (Number of chunks retrieved)', fontsize=14)
    plt.ylabel('Precision (%)', fontsize=14)
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    precision_out = out_dir / "single_run_precision.png"
    plt.savefig(precision_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {precision_out}")

if __name__ == "__main__":
    plot_csv()
