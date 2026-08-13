import matplotlib.pyplot as plt
import os

plt.style.use('bmh')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

os.makedirs('plots', exist_ok=True)

k_levels = [1, 3, 5, 10, 16, 32, 64]

# --- 100 Query Sample Data ---
# Privacy QA
pqa_100_rec = [47.55, 69.75, 80.85, 96.95, 103.55, 105.13, 105.13]
pqa_100_prec = [22.15, 12.51, 9.40, 5.90, 4.19, 2.21, 1.13]
pqa_100_drm = [1.00, 4.33, 7.20, 16.00, 27.69, 60.88, 80.11]

# ContractNLI
cnli_100_rec = [23.41, 31.30, 34.30, 45.42, 51.71, 63.84, 80.84]
cnli_100_prec = [5.19, 2.34, 1.66, 1.12, 0.82, 0.53, 0.35]
cnli_100_drm = [34.00, 60.00, 71.60, 81.90, 87.25, 92.25, 95.00]

# MAUD
maud_100_rec = [8.09, 10.48, 13.08, 21.04, 29.88, 41.93, 51.06]
maud_100_prec = [4.49, 2.01, 1.44, 1.24, 1.22, 0.87, 0.53]
maud_100_drm = [28.24, 34.51, 45.88, 60.12, 66.84, 75.55, 81.45]

# CUAD
cuad_100_rec = [13.68, 47.16, 57.77, 70.34, 77.79, 88.32, 94.23]
cuad_100_prec = [3.80, 4.57, 3.43, 2.15, 1.54, 0.86, 0.47]
cuad_100_drm = [0.00, 1.33, 4.20, 13.30, 21.31, 40.19, 58.59]

# --- Full Corpus Data ---
# ContractNLI
cnli_full_rec = [20.55, 31.81, 35.88, 44.85, 51.32, 62.89, 76.17]
cnli_full_prec = [5.00, 2.74, 1.91, 1.18, 0.84, 0.51, 0.31]

# MAUD
maud_full_rec = [3.84, 6.35, 8.40, 11.26, 14.12, 20.50, 30.84]
maud_full_prec = [2.53, 1.51, 1.25, 0.86, 0.66, 0.49, 0.35]

def plot_100_sample_recall():
    plt.figure(figsize=(10, 6))
    plt.plot(k_levels, pqa_100_rec, marker='o', linewidth=2.5, label='Privacy QA')
    plt.plot(k_levels, cnli_100_rec, marker='s', linewidth=2.5, label='ContractNLI')
    plt.plot(k_levels, maud_100_rec, marker='^', linewidth=2.5, label='MAUD')
    plt.plot(k_levels, cuad_100_rec, marker='d', linewidth=2.5, label='CUAD')
    
    plt.xlabel('Top-K Retrieved Snippets')
    plt.ylabel('Recall (%)')
    plt.title('100-Query Sample: Recall vs Top-K', fontweight='bold')
    plt.ylim(0, 110)
    plt.xlim(0, 65)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/debug_100_recall.png', dpi=300)
    plt.close()

def plot_100_sample_precision():
    plt.figure(figsize=(10, 6))
    plt.plot(k_levels, pqa_100_prec, marker='o', linewidth=2.5, label='Privacy QA')
    plt.plot(k_levels, cnli_100_prec, marker='s', linewidth=2.5, label='ContractNLI')
    plt.plot(k_levels, maud_100_prec, marker='^', linewidth=2.5, label='MAUD')
    plt.plot(k_levels, cuad_100_prec, marker='d', linewidth=2.5, label='CUAD')
    
    plt.xlabel('Top-K Retrieved Snippets')
    plt.ylabel('Precision (%)')
    plt.title('100-Query Sample: Precision vs Top-K', fontweight='bold')
    plt.ylim(0, 30)
    plt.xlim(0, 65)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/debug_100_precision.png', dpi=300)
    plt.close()

def plot_100_sample_drm():
    plt.figure(figsize=(10, 6))
    plt.plot(k_levels, pqa_100_drm, marker='o', linewidth=2.5, label='Privacy QA')
    plt.plot(k_levels, cnli_100_drm, marker='s', linewidth=2.5, label='ContractNLI')
    plt.plot(k_levels, maud_100_drm, marker='^', linewidth=2.5, label='MAUD')
    plt.plot(k_levels, cuad_100_drm, marker='d', linewidth=2.5, label='CUAD')
    
    plt.xlabel('Top-K Retrieved Snippets')
    plt.ylabel('Document Retrieval Miss Rate (DRM %)')
    plt.title('100-Query Sample: DRM vs Top-K', fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(0, 65)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/debug_100_drm.png', dpi=300)
    plt.close()

def plot_full_corpus_recall():
    plt.figure(figsize=(10, 6))
    plt.plot(k_levels, cnli_full_rec, marker='s', linewidth=2.5, color='#e74c3c', label='ContractNLI')
    plt.plot(k_levels, maud_full_rec, marker='^', linewidth=2.5, color='#3498db', label='MAUD')
    
    plt.xlabel('Top-K Retrieved Snippets')
    plt.ylabel('Recall (%)')
    plt.title('Full Corpus: Recall vs Top-K', fontweight='bold')
    plt.ylim(0, 100)
    plt.xlim(0, 65)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/full_corpus_recall.png', dpi=300)
    plt.close()

def plot_full_corpus_precision():
    plt.figure(figsize=(10, 6))
    plt.plot(k_levels, cnli_full_prec, marker='s', linewidth=2.5, color='#e74c3c', label='ContractNLI')
    plt.plot(k_levels, maud_full_prec, marker='^', linewidth=2.5, color='#3498db', label='MAUD')
    
    plt.xlabel('Top-K Retrieved Snippets')
    plt.ylabel('Precision (%)')
    plt.title('Full Corpus: Precision vs Top-K', fontweight='bold')
    plt.ylim(0, 10)
    plt.xlim(0, 65)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/full_corpus_precision.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating Recall/Precision plots...")
    plot_100_sample_recall()
    plot_100_sample_precision()
    plot_100_sample_drm()
    plot_full_corpus_recall()
    plot_full_corpus_precision()
    print("Plots generated successfully in 'plots' directory.")
