# "Kaggle": "https://www.kaggle.com/datasets",
# "UCI Machine Learning Repository": "https://archive.ics.uci.edu/ml/index.php",
# "Google Dataset Search": "https://datasetsearch.research.google.com/",
# "Data.gov": "https://www.data.gov/",
# "World Bank Data": "https://data.worldbank.org/",
# "FiveThirtyEight": "https://data.fivethirtyeight.com/",
# "Reddit Datasets": "https://www.reddit.com/r/datasets/",
# "Awesome Public Datasets": "https://github.com/awesomedata/awesome-public-datasets",
# "Open Data Portal": "https://data.cityofnewyork.us/",  # Example for a local government data portal
# "AWS Public Datasets": "https://registry.opendata.aws/"

import requests
import json
from bs4 import BeautifulSoup

class DataCollector:
    def __init__(self, search_term, theme):
        self.search_term = search_term
        self.theme = theme
        self.url = f"https://en.wikipedia.org/wiki/{self.search_term.replace(' ', '_')}"
        self.raw_data = ""
        self.soup = None

    def fetch_data(self):
        response = requests.get(self.url)
        self.raw_data = response.content
        self.soup = BeautifulSoup(self.raw_data, 'html.parser')
        return self.soup

    def extract_text(self):
        return self.soup.text

    def save_data_as_json(self):
        # Extract the text
        text = self.extract_text()

        # Create a dictionary with the extracted text
        data = {
            "theme": self.theme,
            "search_term": self.search_term,
            "text": text.strip()  # Remove leading/trailing white spaces
        }

        # Generate a filename based on the theme and search term
        filename = f"{self.theme}_{self.search_term.replace(' ', '_')}_data.json"

        # Save data as JSON
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Get user input for the search term and theme
    search_term = input("Enter a search term: ")
    theme = input("Enter a theme for the data: ")

    # Initialize DataCollector with user input
    collector = DataCollector(search_term, theme)
    collector.fetch_data()
    collector.save_data_as_json()

    print(f"Data for '{search_term}' saved as JSON with the theme '{theme}'.")

