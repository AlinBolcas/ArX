import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from utils import output_dir
from modules.tools.text_gen.arx_textGen import librarian, simple_thought

class Knowledge:
    def __init__(self, file_name=str(output_dir() / 'knowledge.json')):
        self.file_name = file_name
        self.default_data_structure = {
            "active_memory": {
                "user": [],
                "arx": [],
                "CO": [],
                "ego": [],
                "superego": []
            },
            "long_term_memory": []
        }
        self.data = self.default_data_structure.copy()
        self.load_from_file()
        self.exchange_counter = 0  # Initialize a counter for exchanges
        self.threshold = 3  # Set a threshold for summarization

    def update_active_memory(self, category, interaction):
        if category in self.data["active_memory"]:
            self.data["active_memory"][category].append(interaction)
            self.exchange_counter += 1
            self.update_long_term_memory(interaction)  # Process for long-term memory
        else:
            print(f"Category '{category}' not found in active memory.")
        self.save_to_file()  # Save after each update

    def get_latest_interactions(self, latest=None):
        if latest is None:
            latest = self.threshold
        # Format active memory for the last 'threshold' number of interactions
        formatted_active_memory = []
        for user_msg, arx_msg in zip(self.data["active_memory"]["user"][-latest:], self.data["active_memory"]["arx"][-latest:]):
            formatted_active_memory.append({"role": "user", "content": user_msg})
            formatted_active_memory.append({"role": "assistant", "content": arx_msg})

        # Get the most recent summary from the long-term memory
        latest_system_memory = self.data["long_term_memory"][-1] if self.data["long_term_memory"] else ""

        # Combine the long-term memory summary with the formatted active memory interactions
        combined_context = {"active_memory": formatted_active_memory, "long_term_memory": latest_system_memory}
        return combined_context

    def update_long_term_memory(self, interaction):
        if self.exchange_counter >= self.threshold:
            # Get the recent interactions for both user and arx
            interactions_to_summarize = {
                "user": self.data["active_memory"]["user"][-self.threshold:],
                "arx": self.data["active_memory"]["arx"][-self.threshold:]
            }
            long_term_memory_data = self.get_latest_system_memory()
                # Handle the case when long_term_memory_data is None
            if long_term_memory_data is None:
                long_term_memory_data = ""

            # Using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(librarian, interactions_to_summarize, long_term_memory_data)
                summary = future.result()  # This will wait until the result is available
            # Update long term memory with the summary
            self.data["long_term_memory"].append(summary)

            # Reset the exchange counter
            self.exchange_counter = 0

            self.save_to_file()
          
    def get_latest_system_memory(self):
        if self.data["long_term_memory"]:
            # Return the latest (last) message from long_term_memory
            return self.data["long_term_memory"][-1]
        else:
            # Return None or an appropriate message if long_term_memory is empty
            return None
    
    def save_to_file(self):
        with open(self.file_name, 'w') as file:
            json.dump(self.data, file, indent=4)

    def reboot_knowledge(self):
        self.data = self.default_data_structure.copy()
        self.save_to_file()
        print("Knowledge has been rebooted to its default structure.")

    def load_from_file(self):
        if os.path.exists(self.file_name):
            try:
                with open(self.file_name, 'r') as file:
                    loaded_data = json.load(file)
                if not all(key in loaded_data for key in self.default_data_structure):
                    print("Knowledge file is missing some keys. Creating a new file.")
                    self.reboot_knowledge()
            except json.JSONDecodeError:
                print("Error reading the knowledge file. Creating a new file.")
                self.reboot_knowledge()
        else:
            self.save_to_file()


# ---------------------------------------------------------
# main for debugging
# ---------------------------------------------------------

def main():
    knowledge_base = Knowledge()  # Automatically loads existing data if present

    output_dir()
    knowledge_base.reboot_knowledge()  # Uncomment to reset the knowledge base
    
    user_prompts = [
        "Hi, I'm Alin and I like art, ice skating, and katanas.",
        "What's my name?",
        "Extrapolate something I might like to know?",
        "tell me a joke.",
        "this was a test.",
        "what did we talk about?.",
        "what are my hobbies and whats my name again?",
    ]

    for prompt in user_prompts:
        
        print(f"user: {prompt}")
        
        knowledge_base.update_active_memory("user", prompt)

        chat_history = knowledge_base.get_latest_interactions()
        response = simple_thought(prompt, chat_history)
        print(f"arx: {response}\n")

        knowledge_base.update_active_memory("arx", response)

if __name__ == "__main__":
    main()
