import requests
import xml.etree.ElementTree as ET

# Quanta Magazine RSS Feed URL
RSS_URL = "https://api.quantamagazine.org/feed/"

def get_greentech_news(limit=5):
    # Define keywords related to greentech to broaden the search
    keywords = ["greentech", "green tech", "climate", "energy", "sustainability", "environment", "solar", "wind"]
    
    try:
        # Fetch the RSS XML data
        response = requests.get(RSS_URL)
        response.raise_for_status()
        
        # Parse the XML
        root = ET.fromstring(response.content)
        
        # Get all items in the feed
        items = root.findall('./channel/item')
        
        print(f"Latest Greentech News from Quanta Magazine:\n")
        
        match_count = 0
        
        # Iterate through all items, not just the first few
        for item in items:
            # Safely extract title and description
            title_elem = item.find('title')
            desc_elem = item.find('description')
            
            title = title_elem.text if title_elem is not None else ""
            description = desc_elem.text if desc_elem is not None else ""
            
            # Combine title and description and convert to lowercase for easy searching
            search_text = (title + " " + description).lower()
            
            # Check if any of our keywords exist in the search text
            if any(keyword in search_text for keyword in keywords):
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                
                # Clean up the date string (remove time for cleaner view)
                short_date = pub_date[:16] if pub_date else "Unknown Date"

                print(f"[{short_date}] {title}")
                print(f"   -> {link}\n")
                
                match_count += 1
                
                # Stop once we hit the requested limit of matched articles
                if match_count >= limit:
                    break
                    
        # Handle the case where no articles match the criteria
        if match_count == 0:
            print(f"No greentech articles found in the current feed.")
            
    except Exception as e:
        print(f"Failed to fetch science news: {e}")

if __name__ == "__main__":
    get_greentech_news()
