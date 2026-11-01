import pandas as pd
import xml.etree.ElementTree as ET
import requests
import time
from datetime import datetime

# Set the pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Configuration Setup
# In OAI-PMH, we fetch the q-bio set directly
OAI_SETS = ["cs"]

START_DATE = "2007-05-23"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# OAI-PMH API Constants
OAI_BASE_URL = "http://export.arxiv.org/oai2"
NS_OAI = "{http://www.openarchives.org/OAI/2.0/}"
NS_ARXIV = "{http://arxiv.org/OAI/arXiv/}"

all_data = []

print(f"Starting historical fetch from {START_DATE} to {END_DATE} using OAI-PMH. This will take some time...")

# 2. Iterate through the major ArXiv sets
for oai_set in OAI_SETS:
    print(f"\n--- Fetching set: {oai_set} ---")
   
    # Initial request URL
    url = f"{OAI_BASE_URL}?verb=ListRecords&set={oai_set}&metadataPrefix=arXiv&from={START_DATE}&until={END_DATE}"
   
    # 3. Paginate through the set using resumptionTokens
    while True:
        # Mandatory polite delay to avoid overwhelming arXiv servers
        time.sleep(5)
       
        try:
            response = requests.get(url)
           
            # The OAI API typically uses 503 errors to enforce rate limits
            if response.status_code == 503:
                retry_after = int(response.headers.get("Retry-After", 30))
                print(f"  -> 503 Service Unavailable. arXiv requests a break. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
               
            response.raise_for_status()
           
        except requests.exceptions.RequestException as e:
            print(f"  -> Network error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue
           
        root = ET.fromstring(response.text)
       
        # Check for OAI-level errors (e.g., no records match the criteria)
        error = root.find(f"{NS_OAI}error")
        if error is not None:
            print(f"  -> API Message: {error.text}")
            break
           
        records = root.findall(f".//{NS_OAI}record")
        print(f"  -> Fetched {len(records)} records in this batch.")
       
        # 4. Parse records (No local category filtering needed, API handles it)
        for record in records:
            metadata = record.find(f"{NS_OAI}metadata")
            if metadata is None:
                continue
           
            arxiv = metadata.find(f"{NS_ARXIV}arXiv")
            if arxiv is None:
                continue
           
            cats = arxiv.find(f"{NS_ARXIV}categories").text if arxiv.find(f"{NS_ARXIV}categories") is not None else ""
           
            # Extract standard fields
            entry_id = arxiv.find(f"{NS_ARXIV}id").text if arxiv.find(f"{NS_ARXIV}id") is not None else None
            title = arxiv.find(f"{NS_ARXIV}title").text.replace('\n', ' ').strip() if arxiv.find(f"{NS_ARXIV}title") is not None else None
            abstract = arxiv.find(f"{NS_ARXIV}abstract").text.replace('\n', ' ').strip() if arxiv.find(f"{NS_ARXIV}abstract") is not None else None
            created = arxiv.find(f"{NS_ARXIV}created").text if arxiv.find(f"{NS_ARXIV}created") is not None else None
            updated = arxiv.find(f"{NS_ARXIV}updated").text if arxiv.find(f"{NS_ARXIV}updated") is not None else None
            doi = arxiv.find(f"{NS_ARXIV}doi").text if arxiv.find(f"{NS_ARXIV}doi") is not None else None
           
            # Extract authors
            authors = [author.find(f"{NS_ARXIV}keyname").text for author in arxiv.findall(f".//{NS_ARXIV}author") if author.find(f"{NS_ARXIV}keyname") is not None]
           
            all_data.append({
                'id': entry_id, 'title': title, 'summary': abstract, 'published': created, 'update': updated,
                'authors': authors, 'doi': doi, 'categories': cats
            })
       
        # 5. Handle Pagination
        token = root.find(f".//{NS_OAI}resumptionToken")
       
        if token is None or not token.text:
            print("  -> End of set.")
            break
           
        # Set up URL for the next iteration using the token
        url = f"{OAI_BASE_URL}?verb=ListRecords&resumptionToken={token.text}"

# 6. Final DataFrame Creation & Export
df = pd.DataFrame(all_data)

# Safety Net: Drop any accidental duplicates based on the unique arXiv ID
df = df.drop_duplicates(subset=['id'], keep='first')

print(f"\n======================================")
print(f"Fetch complete! Total records found: {len(df)}")
print(f"======================================")

# Save to CSV
output_path = 'All_CS.csv'
df.to_csv(output_path, index=False)
print(f"Data saved successfully to {output_path}")
