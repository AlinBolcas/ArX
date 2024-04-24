import os
from dotenv import load_dotenv
import openai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox

openai.api_key = "YOUR_OPENAI_KEY"

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

def gpt4_response(prompt):
    messages=[{"role": "system", "content": 'You are a genius strategist capable of rationally planning expert level flow charts.'}, {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=330,
        temperature=1.2
    )
    return response.choices[0].message['content'].strip()

def gptInstruct_response(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=330,
        temperature=1.2,
    )
    return response.choices[0].text.strip()

def generate_flowchart(topic):
    # Get an overview or breakdown of the topic from GPT
    breakdown = gpt4_response(f"Break down the topic '{topic}' into key points separated by ',':")

    # For simplicity, let's assume the breakdown is a list separated by commas
    points = [point.strip() for point in breakdown.split(',')]

    # Create a new graph
    graph = nx.DiGraph()

    # Add nodes for each point
    for point in points:
        graph.add_node(point)

    # For this simple example, just connect each point sequentially
    for i in range(len(points) - 1):
        graph.add_edge(points[i], points[i+1])

    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, width=3)
    plt.savefig('flowchart.png')
    plt.show()

    return 'flowchart.png'

def draw_graph(graph):
    try:
        pos = nx.drawing.nx_pydot.pydot_layout(graph, prog='dot')
        nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, width=3)
        plt.savefig('flowchart.png')
        plt.show()
    except ImportError:
        print("Graphviz and pydot are required for this layout.")
        # You can fall back to another layout here if you'd like

def main():
    root = tk.Tk()
    root.withdraw() # Hide the main window

    topic = simpledialog.askstring("Input", "What topic would you like to generate a flowchart for?")
    
    if topic:
        generate_flowchart(topic)
        messagebox.showinfo("Info", "Flowchart generated successfully!")

main()
