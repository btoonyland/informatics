import time
import re
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, progress

# 1. Define and expand the keyword lists
ai_keywords = [
    "artificial intelligence", "ai", "machine learning",
    "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "llm"
]

medicine_keywords = [
    "healthcare", "medicine", "health", "hospital", "doctors",
    "clinical", "patient", "medical", "physicians", "nursing"
]

secondary_keywords = [
    "adoption", "diagnostics", "personalize", "operational efficiency",
    "ethical", "legal", "privacy", "cybersecurity", "transparency",
    "accuracy", "strategic implementation", "maturity", "readiness",
    "bias", "compliance", "regulation"
]

def build_regex_pattern(keywords):
    joined_keywords = '|'.join(re.escape(k) for k in keywords)
    return rf'\b({joined_keywords})\b'

ai_pattern = build_regex_pattern(ai_keywords)
med_pattern = build_regex_pattern(medicine_keywords)
sec_pattern = build_regex_pattern(secondary_keywords)

def extract_secondary(matches):
    """Helper function to clean up extracted keywords"""
    if matches is None:
        return []
    # cuDF returns matches as lists; we lowercase and deduplicate
    return list(set(str(match).lower() for match in matches))

def analyze_chunk(df):
    """
    This function processes chunks of the dataframe on the GPU.
    """
    if len(df) > 0:
        sample_text = str(df['summary'].iloc[0])[:80].replace('\n', ' ')
        print(f"[GPU Worker] Processing new batch... Sample: {sample_text}...")

    # 1. Fill NaNs and lowercase
    safe_summary_lower = df['summary'].fillna("").str.lower()

    # 2. Run the GPU regex matching
    ai_mask = safe_summary_lower.str.contains(ai_pattern, regex=True)
    med_mask = safe_summary_lower.str.contains(med_pattern, regex=True)
   
    filtered_df = df[ai_mask & med_mask].copy()
   
    if len(filtered_df) > 0:
        # 3. Extract keywords as raw lists using GPU regex
        safe_filtered_lower = filtered_df['summary'].fillna("").str.lower()
        filtered_df['found_secondary_keywords'] = safe_filtered_lower.str.findall(sec_pattern)
    else:
        filtered_df['found_secondary_keywords'] = None
       
    return filtered_df

if __name__ == '__main__':
    # ==========================================
    # GPU CLUSTER SETUP
    # ==========================================
    # This automatically detects and utilizes your 3 GPUs
    cluster = LocalCUDACluster(n_workers=3)
    client = Client(cluster)
    print(f"Started Dask-CUDA Cluster: {client.dashboard_link}")
    print("Using 3 GPUs for parallel processing...\n")
   
    start_time = time.time()

    # ==========================================
    # DATA LOADING (Distributed)
    # ==========================================
    file_path = "/home/nycticorax/Documents/repos/greentech/All_CS.csv"
    print("Reading CSV into GPU memory...")
   
    # Read directly into a multi-GPU Dask-cuDF dataframe
    ddf = dask_cudf.read_csv(file_path)
   
    # ==========================================
    # PROCESSING
    # ==========================================
    print("\nStarting distributed analysis...")
   
    # map_partitions applies our analysis function across the 3 GPUs in parallel chunks
    # meta defines the expected output schema so Dask can build the computation graph
    meta = ddf._meta.copy()
    meta['found_secondary_keywords'] = 'object'
   
    result_ddf = ddf.map_partitions(analyze_chunk, meta=meta)
   
    # .compute() triggers the actual processing across the GPUs and brings results back to CPU RAM
    result_df = result_ddf.compute()

    # ==========================================
    # RESULTS & TIMING
    # ==========================================
    end_time = time.time()
    elapsed_time = end_time - start_time
   
    print("\n=== RESULTING DATAFRAME METADATA ===")
    result_df.info()
   
    print(f"\n✅ Processing completed in {elapsed_time:.2f} seconds.")

    # Save to Parquet (Highly recommended over CSV for large datasets, especially after using Dask)
    result_df.to_parquet("filtered_ai_medicine_papers_CS.parquet", index=False)
    print("Saved results to filtered_ai_medicine_papers_CSwhy .parquet")
