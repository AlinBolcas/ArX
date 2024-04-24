import os
import requests
from bs4 import BeautifulSoup

def download_image_from_websites(query, websites):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for site in websites:
        try:
            # Construct the search URL for each website
            url = f"https://{site}/search?q={query}"

            # Make the request and parse with BeautifulSoup
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find an image
            img_tag = soup.find("img")

            # If an image is found, download it
            if img_tag:
                img_url = img_tag["src"]
                img_response = requests.get(img_url, stream=True)
                img_response.raise_for_status()

                # Save the image locally
                with open(os.path.join(os.getcwd(), f"{query}.jpg"), 'wb') as img_file:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        img_file.write(chunk)
                print(f"Image for {query} downloaded successfully from {site}!")
                return
        except Exception as e:
            print(f"Error while trying {site}: {e}")
            continue

    print(f"No images found for {query} in the provided websites.")

# List of websites
websites = [
    "artstation.com", "deviantart.com", "unsplash.com", # ... add all the websites here
]

# Test the function
download_image_from_websites('cheese', websites)
