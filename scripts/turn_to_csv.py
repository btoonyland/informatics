import pandas as pd 
df = pd.read_parquet("filtered_ai_medicine_papers.parquet")
df2 = pd.read_parquet("filtered_ai_medicine_papers_CS.parquet")
# This will save a human-readable file you can open in Google Sheets
df.to_csv("filtered_ai_medicine_papers.csv", index=False)
df2.to_csv("filtered_ai_medicine_papers_CS.csv", index=False)
