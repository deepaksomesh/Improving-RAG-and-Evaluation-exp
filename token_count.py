import os
import tiktoken
import glob

def main():
    corpus_dir = r"d:\Thesis\Baselines\summary-augmented-chunking\data\corpus"
    
    # Get all files in corpus_dir (and subdirectories)
    file_paths = glob.glob(os.path.join(corpus_dir, '**', '*.*'), recursive=True)
    # Filter out directories
    file_paths = [f for f in file_paths if os.path.isfile(f)]
    
    if not file_paths:
        print("No files found in corpus directory.")
        return
        
    try:
        # text-embedding-3-small uses cl100k_base
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error getting tokenizer: {e}")
        return
        
    total_tokens = 0
    
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        prepended_text = f"Document Name: {basename}\n\n"
        tokens = encoding.encode(prepended_text)
        total_tokens += len(tokens)
        
    average_tokens = total_tokens / len(file_paths)
    
    print(f"Total documents: {len(file_paths)}")
    print(f"Average tokens for prepended document name: {average_tokens:.2f}")

if __name__ == "__main__":
    main()
