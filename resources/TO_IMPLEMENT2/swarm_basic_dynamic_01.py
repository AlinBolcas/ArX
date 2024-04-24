from multiprocessing import Process, Queue
from queue import Empty
import os
from dotenv import load_dotenv
import openai
import tkinter as tk
from tkinter import simpledialog, messagebox
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pyttsx3

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize task queue and data store
task_queue = Queue()

# Initialize the TTS engine
engine = pyttsx3.init()


# Initialize agents with specific roles and prompts
agents = {
    'historian': 'Provide historical context and lessons regarding {}.',
    'philosopher': 'Provide ethical and philosophical perspectives on {}.',
    'economist': 'Analyze the economic implications of {}.',
    'artist': 'Consider the creative and aesthetic aspects of {}.',
    'scientist': 'Offer empirical evidence and scientific viewpoints on {}.',
    'strategist': 'Offer long-term planning and risk assessment on {}.',
    'futurist': 'Offer predictive and speculative foresight on {}.'
}

def speak(text, rate=250):
    """Speak the given text at the specified rate."""
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

def specialized_agent(q, query, agent_role):
    print(f"Starting {agent_role} agent...")  # Debugging line
    agent_prompt = agents[agent_role].format(query)
    message = [
        {"role": "system", "content": f"You are the {agent_role}. Your role is to {agent_prompt}"},
        {"role": "user", "content": query}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=1.0,
        max_tokens=200,
        frequency_penalty=0.0
    )
    output = response['choices'][0]['message']['content'].strip()
    sentiment_score = analyze_sentiment(output)
    q.put({agent_role: {'output': output, 'sentiment': sentiment_score}})
    print(f"{agent_role} agent finished.")  # Debugging line

def arbitrator_agent(q, data_store):
    print("Starting arbitrator agent...")  # Debugging line
    perspectives = '\n'.join([f"{k.capitalize()}: {v}" for k, v in data_store.items()])
    message = [
        {"role": "system", "content": "arbitrator, jury"},
        {"role": "user", "content": f"Make a synthesis of these perspectives '{perspectives}' into one final generalised insightful statement."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message,
        temperature=1.0,
        max_tokens=500,
        frequency_penalty=0.0
    )
    output = response['choices'][0]['message']['content'].strip()
    q.put({'arbitrator': output})
    print("Arbitrator agent finished.")  # Debugging line
    speak(output)
    

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    query = simpledialog.askstring("Input", "What topic would you like to discuss?")
    
    if query:
        print("Starting specialized agents...")
        processes = [Process(target=specialized_agent, args=(task_queue, query, role)) for role in agents.keys()]
        for p in processes:
            p.start()
            
        data_store = {}
        while len(data_store) < len(agents):
            try:
                result = task_queue.get(timeout=3)
                data_store.update(result)
                print(f"Received result from {list(result.keys())[0]} agent.")
            except Empty:
                print("Waiting for more results...")

        arbitrator_process = Process(target=arbitrator_agent, args=(task_queue, data_store))
        arbitrator_process.start()
        arbitrator_process.join()  # Wait for the arbitrator agent to finish

        arbitrator_result = None
        try:
            arbitrator_result = task_queue.get(timeout=10)
        except Empty:
            print("Waiting for the arbitrator's result...")

        print("Results from specialized agents:")
        for key, value in data_store.items():
            print(f"{key.capitalize()}:\n{value['output']}\n")

        if arbitrator_result:
            print(f"Arbitrator:\n{arbitrator_result['arbitrator']}\n")

        messagebox.showinfo("Info", "Discussion generated successfully!")