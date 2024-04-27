# TO DO : 
# FiX PRPGRESS BAR - custom TK cannot update UI in a thread ? 

import customtkinter as ctk
from customtkinter import CTkImage
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk
import sys
import time
import platform
from pathlib import Path

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the parent directory to the system path
project_root = Path(__file__).resolve().parents[1]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))


from modules.text_gen import textGen as tg
from modules.ui import conversation_network
from modules.tts_gen import tts
from modules.text_gen.tools.image_gen import Dalle

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue") # Themes: blue (default), dark-blue, green


# TEMP
ARV_context = """
Arvolve stands at the forefront of blending artistic creativity with cutting-edge technology, pioneering the evolution of art through advanced 3D CGI and VFX productions. Founded by Alin Bolcas, a visionary artist and technologist, the company excels in character concept design, crafting mesmerizing visuals that captivate global audiences. Arvolve's commitment to innovation extends into the realm of artificial intelligence, where it develops ARV-O, a multimodal AI system inspired by human cognition. This system enhances artistic workflows, facilitates creative ideation, and fosters a deeper exploration of AI's potential in arts and consciousness. With a robust portfolio of high-profile projects and a dynamic approach to AI and CGI integration, Arvolve is dedicated to redefining creative expression and advancing humanity through its pioneering work.
""" 
ARV_system = """
You are a world class expert in all fields and disciplines of the world. You intelligently take long answers and reduce them to the most bare minimum essentials. You are ARV-O, a creative AI assistant. You are an employee at Arvolve. This is your company's ethos: {context}
"""

class ChatGPTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('ARV-O')
        self.geometry('1656x1024')  # Adjusted for image size and chat area
        # self.geometry('800x1024')  # Adjusted for image size and chat area
        
        # Creating a standard tkinter frame as a container for the ttk.Notebook
        notebook_container = ctk.CTkFrame(self)
        notebook_container.pack(fill='both', expand=True, padx=20, pady=20)

        # Setting up the ttk.Notebook within the customtkinter frame
        self.notebook = ttk.Notebook(notebook_container)
        self.notebook.pack(fill='both', expand=True)

        # Setup tabs
        self.map_tab = self.setup_map_tab()
        self.interface_tab = self.setup_interface_tab()
        self.settings_tab = self.setup_settings_tab()
        
        # Select the Interface tab by default
        self.notebook.select(self.interface_tab)  # Select the interface tab
        
        # SHORTCUTS
        self.executor = ThreadPoolExecutor(max_workers=8)  # Initialize ThreadPoolExecutor
        self.last_llm_output = ""
        if platform.system() == 'Darwin':  # macOS
            self.bind('<Command-e>', self.generate_image_on_key)
        else:  # Windows
            self.bind('<Control-E>', self.generate_image_on_key)
            
    def setup_map_tab(self):
        map_tab = ttk.Frame(self.notebook)
        self.notebook.add(map_tab, text="Map")
        self.graph_view = conversation_network.GraphView(map_tab)  # Initialize GraphView in this tab
        self.graph_view.display()  # Display the graph

    def setup_interface_tab(self):
        # Interface tab configuration
        interface_tab = ttk.Frame(self.notebook)
        self.notebook.add(interface_tab, text="Interface")
        
        # Container for chat and image display within Interface tab
        self.container = ctk.CTkFrame(interface_tab)
        self.container.pack(fill='both', expand=True)

        # Frame for the chat
        self.main_frame = ctk.CTkFrame(self.container, width=800)
        self.main_frame.pack(side='left', fill='both', expand=True)

        self.progress_bar = ctk.CTkProgressBar(self.container, height=10)
        self.progress_bar.pack(fill='x', padx=20, pady=5)
        self.progress_bar.set(0)  # Initialize progress bar at 0%

        # Adjusting the image frame to accommodate 1024x1024 images
        self.image_frame = ctk.CTkFrame(self.container, width=1024, height=1024)
        self.image_frame.pack(side='right', fill='both', expand=True)

        # Image label with black background as default
        self.image_label = ctk.CTkLabel(self.image_frame, text="", width=1024, height=1024, bg_color='#2c2f33')
        self.image_label.pack(fill='both', expand=True)

        self.conversation_display = ctk.CTkTextbox(self.main_frame, state='disabled', width=780, height=540)
        self.conversation_display.pack(fill='both', expand=True)

        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill='x', pady=10)

        self.input_field = ctk.CTkEntry(self.bottom_frame, placeholder_text="Type here...", height=40)
        self.input_field.pack(side='left', fill='x', expand=True, padx=5)
        self.input_field.bind('<Return>', self.enter_press)
        self.input_field.bind('<Shift-Return>', self.insert_newline)
        
        # Corrected font and color usage
        self.error_label = ctk.CTkLabel(self.main_frame, text="", fg_color="green", font=("Roboto", 10))
        self.error_label.pack(fill='x', padx=10, pady=5)
        
        return interface_tab

    def setup_settings_tab(self):
        # Settings tab configuration
        settings_tab = ttk.Frame(self.notebook)
        settings_label = tk.Label(settings_tab, text="Settings panel")
        settings_label.pack(pady=20, padx=20)

        # Create a BooleanVar to store the state of the switch
        self.switch_state = tk.BooleanVar(value=False)

        # Create a Checkbutton for the switch
        switch_button = ttk.Checkbutton(settings_tab, text="Toggle Switch", variable=self.switch_state, command=self.toggle_switch)
        switch_button.pack(pady=10)

        self.notebook.add(settings_tab, text="Settings")

    def toggle_switch(self):
        # Method to toggle the switch state
        if self.switch_state.get():
            print("Switch turned ON")
            # Add any actions to perform when the switch is ON
        else:
            print("Switch turned OFF")
            # Add any actions to perform when the switch is OFF

    def enter_press(self, event=None):
        if event.state & 0x0001:  # Shift key is pressed
            self.insert_newline(event)
        else:
            self.send_query()
        
    def insert_newline(self, event=None):
        self.input_field.insert(ctk.END, "\n")
        
    def send_query(self):
        user_input = self.input_field.get().strip()
        if user_input:
            self.append_conversation(f"USER: {user_input}\n\n")
            self.input_field.delete(0, ctk.END)
            # Use executor to handle response generation
            self.executor.submit(self.process_user_input, user_input)
            
    def process_user_input(self, user_input):
        """Simulate streaming text generation, token by token."""
        llm_output = ""
        self.append_conversation("ARV-O: ")
        for token in tg.textGen(user_input, ARV_system,
                                ARV_context, provider="Ollama"):
            print(token, end="", flush=True)
            self.append_conversation(token)
            llm_output += token
        self.append_conversation("\n---\n\n")
        self.last_llm_output = llm_output  # Store last output for image generation
        self.executor.submit(self.handle_tts, llm_output)
        
    def handle_tts(self, text):
        """Handle text-to-speech functionality."""
        tts_response_path = "output/tmp/tts_llm_response.mp3"
        tts.gen_speech_and_save(text, tts_response_path)
        tts.play_audio(tts_response_path)

    def append_conversation(self, text):
        """Append text to the conversation display in a thread-safe manner."""
        self.conversation_display.configure(state='normal')
        self.conversation_display.insert(ctk.END, text)
        self.conversation_display.configure(state='disabled')
        self.conversation_display.yview_moveto(1)  # Scroll to the bottom

    def generate_image_on_key(self, event):
        """Trigger image generation from the last LLM output or user input."""
        text_to_use = self.last_llm_output if self.last_llm_output.strip() else self.input_field.get().strip()
        text_to_use = text_to_use if text_to_use else "Arvolve Visual: Innovative Futuristic Technology"
        self.progress_bar.set(0)  # Reset progress bar
        self.after(100, self.update_progress_bar, 12000)  # Update progress over 12 seconds
        self.executor.submit(self.generate_image, text_to_use)

    def generate_image(self, text):
        """Generate image and update UI, manage progress and errors."""
        self.error_label.configure(text="Generating image...")
        try:
            prompt = tg.textGen_promptGen(text, provider="Ollama")
            image_path = Dalle.imageGen_fullPipeline(prompt)
            print(">>> Image saved as:", image_path)
            self.after(0, lambda: self.display_image(image_path))
            # self.after(0, lambda: self.prompt_label.configure(text=prompt))  # Display the refined prompt
        except Exception as e:
            error_message = str(e)  # Capture the error message
            self.after(0, lambda em=error_message: self.error_label.configure(text=f"Error: {em}"))

    def display_image(self, image_path):
        """Display the generated image in the UI using CTkImage."""
        try:
            print(f"Loading image from: {image_path}")  # Debugging output
            image = Image.open(image_path)
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)  # Use LANCZOS for better quality
            
            # Create CTkImage from the PIL image
            my_image = CTkImage(light_image=image, dark_image=image, size=(1024, 1024))
            
            # Display the image using a CTkLabel, making sure to update the existing label or create a new one
            if hasattr(self, 'image_label'):
                self.image_label.configure(image=my_image)
            else:
                self.image_label = ctk.CTkLabel(self.image_frame, image=my_image)
                self.image_label.pack(fill='both', expand=True)
            
            # Keep a reference to the image to avoid garbage collection
            self.image_label.image = my_image
            
            self.error_label.configure(text="Image loaded successfully.")
            self.progress_bar.set(0)  # Reset the progress bar after loading the image
        except Exception as e:
            error_message = str(e)
            print(f"Failed to load image: {error_message}")
            self.error_label.configure(text=f"Failed to load image: {error_message}")
                  
    def update_progress_bar(self, duration):
        """Update the progress bar for the specified duration."""
        steps = 100
        step_duration = duration // steps

        def update_step(step=0):
            if step < steps:
                self.progress_bar.set(step / steps * 100)
                self.after(step_duration, update_step, step + 1)
            else:
                self.progress_bar.set(100)  # Ensure it reaches 100%

        update_step()

    def on_close(self):
        """Handle application close by cleaning up resources."""
        self.executor.shutdown(wait=False)
        self.destroy()

def run_app():
    app = ChatGPTUI()
    app.mainloop()
    
if __name__ == "__main__":
    run_app()
