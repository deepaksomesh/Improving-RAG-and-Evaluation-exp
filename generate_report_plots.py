import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set professional plotting style
plt.style.use('bmh')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

os.makedirs('plots', exist_ok=True)

def plot_drm_comparison():
    # Data extracted from our ablation experiments
    methods = ['Standard RAG', 'UID Injection', 'Centroid (α=0.5)', 'SAC (Baseline)', 'DocName']
    drm_scores = [21.46, 21.0, 15.75, 4.97, 4.97]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, drm_scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Document Retrieval Miss Rate (DRM %)')
    plt.title('Comparison of DRM Across Methodologies (Lower is Better)', fontweight='bold', pad=20)
    plt.ylim(0, 50)
    
    # Add a horizontal line for the "Solved" threshold
    plt.axhline(y=5.0, color='gray', linestyle='--', alpha=0.7)
    plt.text(-0.4, 5.5, 'SAC Target Baseline', color='gray', fontsize=10, fontstyle='italic')

    plt.tight_layout()
    plt.savefig('plots/drm_comparison.png', dpi=300)
    plt.close()

def plot_centroid_sweep():
    alphas = [1.0, 0.7, 0.5]
    drm = [21.46, 18.20, 15.75]

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, drm, marker='o', linewidth=2.5, markersize=8, color='#f39c12')
    
    for i, txt in enumerate(drm):
        plt.annotate(f'{txt}%', (alphas[i], drm[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.gca().invert_xaxis()  # 1.0 down to 0.5
    plt.xlabel('Alpha (1.0 = No Anchoring, 0.5 = High Anchoring)')
    plt.ylabel('DRM (%)')
    plt.title('Impact of Document Centroid Anchoring on DRM', fontweight='bold')
    plt.ylim(10, 25)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/centroid_sweep.png', dpi=300)
    plt.close()

def plot_mmr_sweep():
    lambdas = [1.0, 0.5]
    drm = [1.97, 43.50]  # Note: Lambda 1.0 DRM was 1.97% because it used DocName as base

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, drm, marker='s', linewidth=2.5, markersize=8, color='#e74c3c')
    
    for i, txt in enumerate(drm):
        plt.annotate(f'{txt}%', (lambdas[i], drm[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.gca().invert_xaxis()  # 1.0 down to 0.5
    plt.xlabel('Lambda (1.0 = Pure Relevance, 0.5 = High Diversity Penalty)')
    plt.ylabel('DRM (%)')
    plt.title('Impact of MMR Diversity Reranking on DocName Baseline', fontweight='bold')
    plt.ylim(0, 50)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/mmr_sweep.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating professional plots for the research report...")
    plot_drm_comparison()
    plot_centroid_sweep()
    plot_mmr_sweep()
    print("Plots saved successfully to the 'plots' directory.")
