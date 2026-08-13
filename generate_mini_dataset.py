import json
import os
import random
import shutil

def main():
    root_dir = "d:/Thesis/Baselines/summary-augmented-chunking"
    orig_corpus_dir = os.path.join(root_dir, "data/corpus")
    orig_bench_dir = os.path.join(root_dir, "data/benchmarks")
    
    out_dir = os.path.join(root_dir, "legalbenchragmini")
    out_corpus_dir = os.path.join(out_dir, "corpus")
    out_bench_dir = os.path.join(out_dir, "benchmark")
    
    os.makedirs(out_corpus_dir, exist_ok=True)
    os.makedirs(out_bench_dir, exist_ok=True)
    
    datasets_target = {
        'privacy_qa': (7, 194),
        'contractnli': (18, 194),
        'maud': (18, 194),
        'cuad': (29, 194)
    }
    
    total_docs = 0
    total_queries = 0
    total_chars = 0
    
    print(f"{'Dataset':<15} {'Docs':<5} {'Queries':<8} {'Chars'}")
    print("-" * 45)
    
    for dataset, (target_docs, target_queries) in datasets_target.items():
        with open(os.path.join(orig_bench_dir, f"{dataset}.json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tests = data['tests']
        
        if dataset == 'privacy_qa':
            # PrivacyQA is inherently 7 docs and 194 queries. No sampling needed.
            selected_tests = tests
            selected_docs = set()
            for t in selected_tests:
                for s in t['snippets']:
                    selected_docs.add(s['file_path'])
        else:
            # Map doc -> queries that ONLY reference that doc
            doc_to_queries = {}
            for t in tests:
                docs_in_test = {s['file_path'] for s in t['snippets']}
                if len(docs_in_test) == 1:
                    doc = list(docs_in_test)[0]
                    if doc not in doc_to_queries:
                        doc_to_queries[doc] = []
                    doc_to_queries[doc].append(t)
            
            all_docs = list(doc_to_queries.keys())
            
            seed = 42
            found_docs = None
            while True:
                random.seed(seed)
                sampled_docs = random.sample(all_docs, target_docs)
                
                gathered_queries = []
                for d in sampled_docs:
                    gathered_queries.extend(doc_to_queries[d])
                    
                if len(gathered_queries) >= target_queries:
                    found_docs = sampled_docs
                    break
                seed += 1
                
            # Now we have found_docs. We need to select exactly target_queries.
            # To ensure every document in found_docs is referenced at least once:
            selected_tests = []
            random.seed(42) # reset seed for reproducibility of query selection
            
            # Step 1: Pick 1 random query per doc
            for d in found_docs:
                q = random.choice(doc_to_queries[d])
                selected_tests.append(q)
                
            # Step 2: Pick remaining queries from the pooled set (excluding the ones already picked)
            remaining_pool = [q for d in found_docs for q in doc_to_queries[d] if q not in selected_tests]
            random.shuffle(remaining_pool)
            
            needed = target_queries - len(selected_tests)
            selected_tests.extend(remaining_pool[:needed])
            
            # Final validation check
            selected_docs = set()
            for t in selected_tests:
                for s in t['snippets']:
                    selected_docs.add(s['file_path'])
                    
            assert len(selected_docs) == target_docs
            assert len(selected_tests) == target_queries

        # Save Benchmark JSON
        data['tests'] = selected_tests
        with open(os.path.join(out_bench_dir, f"{dataset}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        # Copy Corpus Files and calculate characters
        dataset_chars = 0
        for doc_path in selected_docs:
            src = os.path.join(orig_corpus_dir, doc_path)
            dst = os.path.join(out_corpus_dir, doc_path)
            
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(src):
                print(f"MISSING: {src}")
            else:
                shutil.copy2(src, dst)
                
            try:
                with open(dst, 'r', encoding='utf-8') as f:
                    dataset_chars += len(f.read())
            except Exception as e:
                pass

                
        print(f"{dataset:<15} {len(selected_docs):<5} {len(selected_tests):<8} {dataset_chars:,}")
        
        total_docs += len(selected_docs)
        total_queries += len(selected_tests)
        total_chars += dataset_chars

    print("-" * 45)
    print(f"{'Total':<15} {total_docs:<5} {total_queries:<8} {total_chars:,}")
    print("\nSuccessfully generated LegalBench-RAG-mini dataset in /legalbenchragmini!")

if __name__ == "__main__":
    main()
