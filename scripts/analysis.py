import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans

# ==========================================
# 1. SETUP & TEXT CLEANING
# ==========================================
# Load your filtered dataset
df = pd.read_parquet("filtered_ai_medicine_papers_CS.parquet")
print(f"Loaded {len(df)} papers for analysis.\n")

# Define useless academic filler words
academic_stop_words = {
    'paper', 'study', 'propose', 'proposed', 'results', 'show', 'showed',
    'research', 'article', 'conclusion', 'based', 'approach', 'method',
    'model', 'models', 'using', 'used', 'new', 'data', 'analysis'
}

# Combine standard English stop words with our custom academic ones
custom_stop_words = list(ENGLISH_STOP_WORDS.union(academic_stop_words))

def clean_academic_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'\d+', '', text)               # Remove all numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)      # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra white spaces
    return text

print("Cleaning text and removing academic filler words...")
df['clean_summary'] = df['summary'].fillna("").apply(clean_academic_text)
cleaned_summaries = df['clean_summary']
print("="*50)


# ==========================================
# 2. N-GRAM ANALYSIS (Most Common Phrases)
# ==========================================
print("Extracting top 100 n-grams...")

count_vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=custom_stop_words, max_features=2000)
ngram_matrix = count_vectorizer.fit_transform(cleaned_summaries)

phrase_frequencies = sum(ngram_matrix).toarray()[0]
phrases = count_vectorizer.get_feature_names_out()

phrase_freq_dict = [(phrases[i], phrase_frequencies[i]) for i in range(len(phrases))]
phrase_freq_dict.sort(key=lambda x: x[1], reverse=True)

with open("./analysis/ngram_results_CS.txt", "w", encoding="utf-8") as file:
    file.write("TOP 100 MOST COMMON PHRASES\n")
    file.write("===========================\n")
    for phrase, count in phrase_freq_dict[:100]:
        file.write(f"{count} occurrences: {phrase}\n")

print("✅ Saved top 100 n-grams to 'ngram_results.txt'")
print("="*50)


# ==========================================
# 3. TOPIC CLUSTERING & KEYWORDS
# ==========================================
NUM_CLUSTERS = 5
print(f"Clustering papers into {NUM_CLUSTERS} distinct topics...\n")

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=custom_stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_summaries)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

df['topic_cluster'] = kmeans.labels_
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()

cluster_keywords_dict = {}

with open("cluster_keywords.txt", "w", encoding="utf-8") as file:
    file.write("CLUSTER TOPICS & KEYWORDS\n")
    file.write("=========================\n\n")
   
    for i in range(NUM_CLUSTERS):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        keywords_str = ", ".join(top_words)
        cluster_keywords_dict[i] = keywords_str
        cluster_size = (df['topic_cluster'] == i).sum()
       
        file.write(f"Cluster {i} ({cluster_size} papers):\n")
        file.write(f"Keywords: {keywords_str}\n\n")

print("✅ Saved cluster keywords to 'cluster_keywords.txt'")

# Attach the keywords to the dataset so they are visible in the CSV
df['cluster_keywords'] = df['topic_cluster'].map(cluster_keywords_dict)

# Save the final dataset as a CSV so you can open it in Google Sheets
df.to_csv("./analysis/clustered_ai_medicine_papers_CS.csv", index=False)
print("✅ Saved clustered dataset to 'clustered_ai_medicine_papers.csv'")
