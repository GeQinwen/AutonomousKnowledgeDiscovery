"""Download all SciSciNet v2 parquet files from HuggingFace."""

import os
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Northwestern-CSSI/sciscinet-v2"
LOCAL_DIR = "./data/sciscinet_v2"

# All parquet files, ordered small → large
FILES = [
    # Tiny (< 1 MB)
    "sciscinet_fields.parquet",                  # 41.2 KB
    "sciscinet_affl_assoc_affl.parquet",         # 313 KB
    "sciscinet_link_nobellaureates.parquet",     # 482 KB
    # Small (1-100 MB)
    "sciscinet_link_clinicaltrials.parquet",     # 4.57 MB
    "sciscinet_affiliations.parquet",            # 5.27 MB
    "sciscinet_sources.parquet",                 # 8.9 MB
    "sciscinet_nih_metadata.parquet",            # 9.58 MB
    "sciscinet_nsf_metadata.parquet",            # 9.27 MB
    "sciscinet_link_nsf.parquet",                # 16.6 MB
    "sciscinet_link_newsfeed.parquet",           # 34 MB
    "sciscinet_clinicaltrials_metadata.parquet", # 34.2 MB
    "sciscinet_link_nih.parquet",                # 51.5 MB
    "sciscinet_newsfeed_metadata.parquet",       # 74.4 MB
    # Medium (100 MB - 1 GB)
    "sciscinet_papers_pmid_pmcid.parquet",       # 156 MB
    "sciscinet_link_patents.parquet",            # 330 MB
    # Large (1-10 GB)
    "sciscinet_authors.parquet",                 # 2.39 GB
    "sciscinet_papers.parquet",                  # 5.26 GB
    "hit_papers_level0.parquet",                 # 2.69 GB
    "hit_papers_level1.parquet",                 # 2.71 GB
    "sciscinet_author_details.parquet",          # 4 GB
    "normalized_citations_level0.parquet",       # 4.23 GB
    "sciscinet_papersources.parquet",            # 4.38 GB
    "normalized_citations_level1.parquet",       # 5.85 GB
    "sciscinet_paper_author_affiliation.parquet",# 6.86 GB
    # Very large (> 10 GB)
    "sciscinet_paperfields.parquet",             # 10.6 GB
    "sciscinet_authors_paperid.parquet",         # 12.1 GB
    "sciscinet_paperrefs.parquet",               # 19.5 GB
]


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    total = len(FILES)
    t0 = time.time()

    for i, fname in enumerate(FILES, 1):
        dest = Path(LOCAL_DIR) / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i}/{total}] SKIP (exists): {fname}")
            continue

        print(f"[{i}/{total}] Downloading {fname} ...")
        t1 = time.time()
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
            )
            elapsed = time.time() - t1
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK: {size_mb:.1f} MB in {elapsed:.0f}s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    total_elapsed = time.time() - t0
    print(f"\nDone. Total time: {total_elapsed/60:.1f} minutes")

    # Summary
    print("\nFiles in", LOCAL_DIR, ":")
    total_size = 0
    for f in sorted(Path(LOCAL_DIR).glob("*.parquet")):
        sz = f.stat().st_size / (1024**3)
        total_size += sz
        print(f"  {f.name:50s} {sz:.2f} GB")
    print(f"  {'TOTAL':50s} {total_size:.1f} GB")


if __name__ == "__main__":
    main()
