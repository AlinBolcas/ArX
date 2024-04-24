import os
from dotenv import load_dotenv
import openai
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx
import tkinter as tk
from tkinter import simpledialog, messagebox
import re

import sys

# Set a higher recursion depth limit
sys.setrecursionlimit(50)  # You can adjust the value as needed

# Initialize OpenAI GPT-4
def initialize_gpt():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key

# Function to Process GPT-4 Response
def process_gpt_response(response):
    return re.findall(r'\b\w+\b', response)

def gpt_response(prompt):
    model_engine = "text-davinci-002"
    example_prompt = f"{prompt}. List the related concepts separated by commas:"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=example_prompt,
        max_tokens=100
    )
    concepts = response['choices'][0]['text'].split(", ")
    print("Related concepts fetched: ", concepts)
    return concepts

def convert_graph_to_data(graph, num_features):
    # Extract nodes and edges from the graph
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    # Create a PyTorch Geometric Data object
    x = torch.zeros(len(nodes), num_features)  # Create node features
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)

    return data


# Set Placeholder Training Data
def set_training_data(data, num_classes):
    data.y = torch.tensor([i % num_classes for i in range(data.num_nodes)], dtype=torch.long)
    data.train_mask = torch.tensor([True if i % 2 == 0 else False for i in range(data.num_nodes)], dtype=torch.bool)
    return data

# GNN Definition
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

visited_nodes = set()
max_nodes = 10  # set a reasonable number
information_threshold = 0.1  # set according to your specific need

def compute_information_value(concept, node):
    # Your logic to compute info_value using both concept and node
    return len(concept) + len(node)


def recursive_expand(node, graph, depth=0, max_depth=3, info_value=1.0):
    global visited_nodes
    print(f"Expanding node: {node}, Depth: {depth}, Information Value: {info_value}")

    if (
        depth > max_depth
        or node in visited_nodes
        or len(graph.nodes) >= max_nodes
        or info_value < information_threshold
    ):
        return False  # Terminate expansion

    visited_nodes.add(node)
    related_concepts = gpt_response(f"Expand the concept {node}")

    for concept in related_concepts:
        new_info_value = compute_information_value(concept, node)
        
        # Ensure valid integer pairs in edges
        if isinstance(node, int) and isinstance(concept, int):
            graph.add_edge(node, concept, weight=new_info_value)  # Add an edge from node to concept
        
        if recursive_expand(concept, graph, depth + 1, max_depth, new_info_value):
            return True  # Continue expansion

    return False  # Terminate expansion



# Placeholder function for decision-making
def gnn_decision_function(out, graph):
    # Modify this function based on your decision-making logic
    # For now, it just returns the first node in the graph as a placeholder.
    return list(graph.nodes)[0]


# Main Function
def main():
    root = tk.Tk()
    root.withdraw()
    topic = simpledialog.askstring("Input", "What topic would you like to generate a flowchart for?")
    if topic:
        initialize_gpt()
        graph = nx.DiGraph()
        graph.add_node(topic)

        # Add a counter for iterations
        iterations = 0
        
        while recursive_expand(topic, graph):
            iterations += 1
            if iterations >= max_iterations:
                break
        
        num_features = 10  # Define num_features here (or dynamically determine it)

        data = convert_graph_to_data(graph, num_features)  # Pass num_features to the function

        num_classes = 3  # Or dynamically determine this based on your needs

        model = Net(num_features, num_classes)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(200):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
            loss.backward()
            optimizer.step()

        decision_node = gnn_decision_function(out, graph)  # Pass the 'graph' variable
        final_response = gpt_response(f"Write about {decision_node}")

        print(final_response)
        messagebox.showinfo("Info", "Flowchart generated successfully!")

if __name__ == "__main__":
    main()
