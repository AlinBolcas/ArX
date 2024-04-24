import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np



class GraphView:
    def __init__(self, master):
        self.master = master
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)  # Create the canvas

        plt.style.use('dark_background')

        # Make background transparent
        self.fig.patch.set_alpha(0.0)
        self.ax.patch.set_alpha(0.0)
        
        # Set figure and axes background color explicitly
        # self.ax.set_facecolor('#2c2f33')  # Dark grey background
        # self.fig.patch.set_facecolor('#2c2f33')  # Matching figure background
        
        # self.ax.set_facecolor("yellow")
        # Explicitly draw the canvas first to apply initial settings
        self.canvas.draw()

        self.setup_graph()

    def setup_graph(self):
        # Create a graph with some example nodes and edges
        self.G = nx.Graph()
        self.G.add_edge('conv1', 'conv2')
        self.G.add_edge('conv2', 'conv3')
        self.G.add_edge('conv3', 'conv1')

        # Layout and draw the graph
        self.positions = nx.spring_layout(self.G)
        nx.draw(self.G, pos=self.positions, ax=self.ax, with_labels=True,
                node_size=1000, node_color='#2c2f33', 
                font_size=8, font_color='red',
                font_weight='bold', edge_color='black')

        self.canvas.draw()  # Redraw the canvas to apply all changes
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)  # Connect click event


    def on_click(self, event):
        if event.inaxes is not None:
            click_coords = np.array([event.xdata, event.ydata])  # Click coordinates as a NumPy array
            node_distances = []

            for node, pos in self.positions.items():
                node_pos = np.array(pos)  # Convert the position to a NumPy array
                distance = np.linalg.norm(node_pos - click_coords)  # Calculate Euclidean distance
                node_distances.append((node, distance))

            # Find the closest node by sorting the distances
            closest_node, min_distance = min(node_distances, key=lambda x: x[1])
            
            if min_distance < 0.1:  # Threshold for closeness
                print(f"Node {closest_node} clicked")
                # Add any additional functionality here for when a node is clicked

    def display(self):
        # Pack the canvas to the Tkinter window
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Graph View Standalone Test")
    root.geometry("800x600")
    app = GraphView(root)
    app.display()
    root.mainloop()
