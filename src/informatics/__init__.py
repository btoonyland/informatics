"""Harvest arXiv metadata, select AI-in-medicine papers, and analyse them.

Pipeline (each step is an ``informatics`` sub-command, see ``informatics --help``):

1. ``harvest`` / ``import-kaggle`` / ``import-legacy-csv``  -> data/raw/<set>.parquet
2. ``filter``   -> data/interim/<set>_ai_medicine.parquet (+ AI-only background sample)
3. ``analyze``  -> results/<set>/ (phrases, clusters, themes, report.md)
"""

__version__ = "0.1.0"
