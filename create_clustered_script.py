import os

in_path = 'd:/Thesis/Baselines/summary-augmented-chunking/benchmarks/legalbenchrag/run_benchmark_mini_centroid.py'
out_path = 'd:/Thesis/Baselines/summary-augmented-chunking/benchmarks/legalbenchrag/run_benchmark_mini_clustered_centroid.py'

with open(in_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Replace class name
code = code.replace('class CentroidBaselineRetrievalMethod', 'class ClusteredCentroidBaselineRetrievalMethod')
code = code.replace('CentroidBaselineRetrievalMethod', 'ClusteredCentroidBaselineRetrievalMethod')

# Replace prints
code = code.replace('CentroidBaseline:', 'ClusteredCentroidBaseline:')
code = code.replace('Legalbench-RAG Centroid benchmark', 'Legalbench-RAG Clustered Centroid benchmark')
code = code.replace('legalbenchrag_centroid', 'legalbenchrag_clustered_centroid')
code = code.replace('CentroidBaseline (alpha', 'ClusteredCentroidBaseline (alpha')

old_logic = """        # 2. Compute Centroids
        doc_indices = {}
        for i, chunk in enumerate(all_chunks):
            if chunk.file_path not in doc_indices:
                doc_indices[chunk.file_path] = []
            doc_indices[chunk.file_path].append(i)

        centroids = np.zeros_like(emb_np)
        for doc_id, indices in doc_indices.items():
            doc_embs = emb_np[indices]
            centroid = np.mean(doc_embs, axis=0)
            
            # Normalize centroid
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
                
            centroids[indices] = centroid"""

new_logic = """        from sklearn.cluster import KMeans
        import warnings
        os.environ["OMP_NUM_THREADS"] = "1"
        warnings.filterwarnings("ignore", category=UserWarning)

        # 2. Compute Clustered Centroids
        doc_indices = {}
        for i, chunk in enumerate(all_chunks):
            if chunk.file_path not in doc_indices:
                doc_indices[chunk.file_path] = []
            doc_indices[chunk.file_path].append(i)

        centroids = np.zeros_like(emb_np)
        for doc_id, indices in doc_indices.items():
            doc_embs = emb_np[indices]
            n_chunks = len(doc_embs)
            k = max(1, n_chunks // 5)  # 1 cluster per 5 chunks
            
            if k == 1 or n_chunks < 2:
                centroid = np.mean(doc_embs, axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 0:
                    centroid = centroid / c_norm
                centroids[indices] = centroid
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(doc_embs)
                cluster_centers = kmeans.cluster_centers_
                
                c_norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
                c_norms[c_norms == 0] = 1e-10
                cluster_centers = cluster_centers / c_norms
                
                for local_idx, global_idx in enumerate(indices):
                    cluster_label = labels[local_idx]
                    centroids[global_idx] = cluster_centers[cluster_label]"""

if old_logic in code:
    code = code.replace(old_logic, new_logic)
else:
    print('Error: Could not find old centroid logic to replace!')

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(code)
    print("Successfully created the clustered centroid script.")
