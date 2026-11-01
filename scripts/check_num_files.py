import requests
import xml.etree.ElementTree as ET
import time

# Query for all q-bio categories.
# Note: The standard API handles dates slightly differently, so this gets the all-time total.
url = "http://export.arxiv.org/api/query?search_query=cat:q-bio*&start=0&max_results=0"

response = requests.get(url)
try:
    root = ET.fromstring(response.text)
except ET.ParseError:
    print(f"  -> XML Parse Error: Server did not return valid XML.")
    print(f"  -> Server response preview: {response.text[:250]}...")
    print("  -> Waiting 10 seconds and retrying...")
    time.sleep(10)

# The arXiv API uses the opensearch namespace for result counts
ns = {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
total_results = root.find('opensearch:totalResults', ns).text

print(f"Total all-time q-bio articles available: {total_results}")
