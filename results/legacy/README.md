# Legacy results (first version of the pipeline)

These files were produced by the old `scripts/` pipeline, which this repository has replaced. They are
kept for reference, but **do not use them for conclusions**. Re-run the current pipeline instead
(see the main README). Known problems:

**Data**

- The two `clustered_*.csv` files are 9-row samples of the full output, not the complete results.
- 11 of the 18 arXiv IDs are damaged. They were read back from CSV as numbers, losing leading and
  trailing zeros (`0708.2931` → `708.2931`, `1004.2280` → `1004.228`).
  `informatics.arxiv_ids.normalize_id` repairs them.
- Accented author names are garbled (`Zukić` → `ZukiÄ\x87`): the harvester decoded arXiv's UTF-8
  as Latin-1. Only surnames were kept.
- `published` is not the first-submission date: in 7 of 18 rows it is 2–14 months later than the
  month encoded in the ID, and `update` is `2009-12-01` for 7 rows.
- `found_secondary_keywords` contains duplicates and is written as a numpy repr
  (`['accuracy' 'accuracy']`), which Python parses back as `['accuracyaccuracy']`.
- `1004.2280` appears in both the cs and q-bio files. The two sets overlap through cross-listing
  but were analysed separately.

**Phrase counts (`ngram_results*.txt`)**

- Stop words were deleted *before* phrases were formed, so the lists contain phrases that never
  occur in the text: "state art", "end end", "chest ray" (x-ray), "large language llms" (models).
  The custom stop list also removed `data`, `model(s)` and `analysis`, and scikit-learn's list
  removed words such as `side`, `back` and `interest`.
- Digits and accented letters were stripped first ("SARS-CoV-2" → "sars cov", "U-Net" → "net").
- Counts are mentions, not papers, and are not normalised. The cs corpus is about ten times
  larger than q-bio, so the counts cannot be compared.
- 6 (q-bio) and 10 (cs) of the top 100 entries are fragments of a longer entry with almost the
  same count ("intelligence ai" ⊂ "artificial intelligence ai"). Singular and plural forms are
  listed separately.
- Half or more of the top 10 phrases contain the keywords the papers were selected on.

**Clusters**

- The number of clusters was fixed at 5. 3–6 of each cluster's 10 label words are the selection
  keywords themselves.
