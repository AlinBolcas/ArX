import requests
import json
from bs4 import BeautifulSoup
import openai
import os
import pyttsx3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the TTS engine
engine = pyttsx3.init()

def speak(text, rate=200):
    """Speak the given text at the specified rate."""
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

class WikipediaDataCollector:
    def __init__(self):
        self.search_term = ""
        self.raw_data = ""
        self.soup = None

    def fetch_data(self):
        wikipedia_url = f"https://en.wikipedia.org/wiki/{self.search_term.replace(' ', '_')}"
        response = requests.get(wikipedia_url)
        if response.status_code == 200:
            self.raw_data = response.content
            self.soup = BeautifulSoup(self.raw_data, 'html.parser')
            return self.soup
        else:
            print(f"Error fetching data from Wikipedia for '{self.search_term}': Status Code {response.status_code}")
            return None

    def extract_text(self):
        return self.soup.text

    def save_data_as_json(self):
        text = self.extract_text()
        # Replace newline characters with actual newlines for readability
        text = text.replace('\n', '\n')
        data = {
            "search_term": self.search_term,
            "text": text.strip()
        }
        filename = f"Wikipedia_{self.search_term.replace(' ', '_')}_data.json"
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    def read_data_with_gpt35(self):
        filename = f"Wikipedia_{self.search_term.replace(' ', '_')}_data.json"
        with open(filename, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            data_to_synthesize = json_data["text"]

        # Split the text into smaller chunks while maintaining sentences
        text_chunks = []
        current_chunk = ""
        sentences = data_to_synthesize.split('. ')  # Split by sentences
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 4000:
                current_chunk += sentence + '. '
            else:
                text_chunks.append(current_chunk)
                current_chunk = sentence + '. '

        # Process each chunk and generate responses
        synthesized_data = []
        for chunk in text_chunks:
            response = openai.Completion.create(
                engine="text-davinci-002",  # Use GPT-3.5 Turbo
                prompt=f"Summarize the following text as part of a larger corpus: '{chunk}' which is about '{self.search_term}'",
                max_tokens=350,
                temperature=0.7
            )
            synthesized_data.append(response.choices[0].text.strip())

        synthesized_text = " ".join(synthesized_data)
        speak(synthesized_text, rate=300)
        return synthesized_text

if __name__ == "__main__":
    print("Wikipedia Data Collection and Synthesis")
    
    search_term = input("Enter a search term for Wikipedia: ")

    collector = WikipediaDataCollector()
    collector.search_term = search_term

    collector.fetch_data()
    collector.save_data_as_json()
    synthesized_data = collector.read_data_with_gpt35()

    print(f"Data for '{search_term}' from Wikipedia saved as JSON.")
    print(f"Synthesized Data: {synthesized_data}")
