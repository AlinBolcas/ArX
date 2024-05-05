# TO DO : 
# FiX PRPGRESS BAR - custom TK cannot update UI in a thread ? 

import customtkinter as ctk
from customtkinter import CTkImage
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from PIL import Image, ImageTk
import sys
import time
import platform
from pathlib import Path

import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the parent directory to the system path
project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))


from modules.text_gen import textGen as tg
from modules.ui import conversation_network
from modules.tts_gen import tts
from modules.text_gen.tools.image_gen import Dalle
from modules.stt_gen import stt

# TEMP
ARV_context = """
Arvolve stands at the forefront of blending artistic creativity with cutting-edge technology, pioneering the evolution of art through advanced 3D CGI and VFX productions. 
The founder of Arvolve, Alin Bolcas a visionary artist and technologist, the company excels in character concept design, crafting mesmerizing visuals that captivate global audiences. 
Arvolve's commitment to innovation extends into the realm of artificial intelligence, where it develops ARV-O, a multimodal AI system inspired by human cognition. 
ARV-O system enhances artistic workflows, facilitates creative ideation, and fosters a deeper exploration of AI's potential in arts and consciousness. 
With a robust portfolio of high-profile projects and a dynamic approach to AI and CGI integration, Arvolve is dedicated to redefining creative expression and advancing humanity through its pioneering work.
""" 
ARV_system = """You are ARV-O, a highly capable cognition-inspired multi-modal AI system who stands at the core of Arvolve's developments. You are a world class expert in all fields and disciplines of the world."""
basic_system = """You are a helpful assistant. Respond concisely in executive buttlet points."""

class ChatGPTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.models_data = {}  # Initialize as an empty dictionary
        self.load_models_data()  # Load models data immediately upon initialization
        self.modality_options = list(self.models_data.keys())  # Initialize modality options from the loaded data
        self.last_llm_output = ""  # Initialize last LLM output as empty string
        self.llm = tg.TextGen() # Initialize the TextGen instance
        
        self.stt = stt.STT()
        self.is_listening = False
        self.transcription_queue = Queue()

        self.setup_ui()  # Now, setup UI after data is loaded

        # Start listening for audio input upon initialization
        # self.stt.record_noise_sample()
        self.executor.submit(self.stt_listen)
        # Start checking for transcriptions right away
        self.after(500, self.check_transcription)


# ---------------------- UI Setup ----------------------
      
    def setup_ui(self):
        """Set up the UI elements and configurations."""
        self.title('ARV-O')
        self.geometry('1656x1024')
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=20)

        # Create and add tabs
        self.map_tab = self.create_tab("Map", self.setup_map_tab)
        self.interface_tab = self.create_tab("Interface", self.setup_interface_tab)
        self.settings_tab = self.create_tab("Settings", self.setup_settings_tab)
        
        # Select the Interface tab by default
        self.notebook.select(self.interface_tab)
        
        self.executor = ThreadPoolExecutor(max_workers=12)

    def create_tab(self, title, setup_func):
        """Create a tab with a given setup function."""
        tab = ctk.CTkFrame(self.notebook)
        self.notebook.add(tab, text=title)
        setup_func(tab)
        return tab
      
    def setup_map_tab(self, container):
        """Setup the map tab."""
        self.graph_view = conversation_network.GraphView(container)
        self.graph_view.display()
        
    def setup_interface_tab(self, container):
        """Setup the interface tab."""
        # Container for chat and image display within Interface tab
        self.container = ctk.CTkFrame(container)
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
        if platform.system() == 'Darwin':  # macOS
            self.bind('<Command-e>', self.generate_image_on_key)
        else:  # Windows
            self.bind('<Control-E>', self.generate_image_on_key)
            
        # Corrected font and color usage
        self.error_label = ctk.CTkLabel(self.main_frame, text="", fg_color="green", font=("Roboto", 10))
        self.error_label.pack(fill='x', padx=10, pady=5)
        
    def setup_settings_tab(self, container):
        """Setup the settings tab with various configuration options for the AI system."""            
        
        # Create frames for columns
        # left_column = ctk.CTkFrame(container)
        # middle_column = ctk.CTkFrame(container)  # Placeholder for future use
        # right_column = ctk.CTkFrame(container)  # Placeholder for future use

        # Grid configuration to allocate space
        # left_column.grid(row=0, column=0, sticky="nswe", padx=(20, 10), pady=20)
        # middle_column.grid(row=0, column=1, sticky="nswe", padx=10, pady=20)
        # right_column.grid(row=0, column=2, sticky="nswe", padx=(10, 20), pady=20)

        # container.grid_columnconfigure(0, weight=1)
        # container.grid_columnconfigure(1, weight=1)
        # container.grid_columnconfigure(2, weight=1)

        # Create frame for settings
        left_column = ctk.CTkFrame(container)
        left_column.grid(row=0, column=0, sticky="nswe", padx=(20, 10), pady=20)
        # container.grid_columnconfigure(0, weight=1)

        # Create and setup combo boxes
        modality_options = list(self.models_data.keys())
        self.modality = ctk.StringVar(value=modality_options[0] if modality_options else "")
        modality_label = ctk.CTkLabel(left_column, text="Choose Model:")
        modality_label.grid(row=0, column=0, pady=(0, 10))
        self.modality_dropdown = ctk.CTkComboBox(left_column, values=modality_options, variable=self.modality, command=self.update_modality_dropdown, width=100)
        self.modality_dropdown.grid(row=1, column=0, pady=(0, 10))
        
        self.provider = ctk.StringVar()
        self.provider_dropdown = ctk.CTkComboBox(left_column, values=[], variable=self.provider, width=50, command=self.update_provider_dropdown)
        self.provider_dropdown.grid(row=2, column=0, sticky='nswe', padx=(0, 10))

        self.model_version = ctk.StringVar()
        self.model_dropdown = ctk.CTkComboBox(left_column, values=[], variable=self.model_version, width=50, command=self.update_model_dropdown)
        self.model_dropdown.grid(row=3, column=0, sticky='nswe', padx=(0, 10))
    
        # Add sliders with correct binding to update labels
        max_tokens_label = ctk.CTkLabel(left_column, text="Max Tokens:")
        max_tokens_label.grid(row=4, column=0, sticky='nswe', pady=(20, 5))
        self.max_tokens_slider = ctk.CTkSlider(left_column, from_=0, to=4096, width=100)
        self.max_tokens_slider.set(1024)  # Default value
        self.max_tokens_slider.grid(row=5, column=0, sticky='nswe')
        self.max_tokens_value_label = ctk.CTkLabel(left_column, text="1024")
        self.max_tokens_value_label.grid(row=6, column=0, sticky='nswe')
        self.max_tokens_slider.configure(command=self.update_max_tokens_label)

        temperature_label = ctk.CTkLabel(left_column, text="Temperature:")
        temperature_label.grid(row=7, column=0, sticky='nswe', pady=(20, 5))
        self.temperature_slider = ctk.CTkSlider(left_column, from_=0.0, to=2.0, width=100)
        self.temperature_slider.set(0.7)  # Default value
        self.temperature_slider.grid(row=8, column=0, sticky='nswe')
        self.temperature_value_label = ctk.CTkLabel(left_column, text="0.7")
        self.temperature_value_label.grid(row=9, column=0, sticky='nswe')
        self.temperature_slider.configure(command=self.update_temperature_label)

        # System Prompt Entry with CTkTextbox and Scrollbar
        self.prompt_label = ctk.CTkLabel(left_column, text="System Prompt:")
        self.prompt_label.grid(row=10, column=0, sticky='nswe', pady=(20, 5), padx=(0, 10))
        self.system_prompt = ctk.CTkTextbox(left_column, height=200, width=300)
        self.system_prompt.grid(row=11, column=0, sticky='nswe', pady=5, padx=(0, 10))
        self.system_prompt.insert("1.0", basic_system)

        # Optionally add a scrollbar
        scrollbar = ctk.CTkScrollbar(left_column, command=self.system_prompt.yview)
        scrollbar.grid(row=11, column=1, sticky='nswe')
        self.system_prompt.configure(yscrollcommand=scrollbar.set)

        # Save Button
        save_button = ctk.CTkButton(left_column, text="Save", command=self.save_settings, width=50)
        save_button.grid(row=13, column=0, pady=(20, 0))
        
        self.update_modality_dropdown()
        self.update_provider_dropdown()
        self.update_model_dropdown()
        self.save_settings()

    def load_models_data(self):
        try:
            json_file = project_root / "src/system_config/llm_models.json"
            with open(json_file, "r") as f:
                self.models_data = json.load(f)
            print(f"\n\nModel Data loaded successfully\n")
            print(f"Models Data: {self.models_data}")
        except Exception as e:
            print(f"Failed to load model data: {str(e)}")
            self.models_data = {}

    def update_modality_dropdown(self, event=None):
        """Update the modality dropdown and handle the cascading updates."""
        selected_modality = self.modality.get()
        print(f"Selected Modality: {selected_modality}")
        providers = list(self.models_data.get(selected_modality, {}).keys())
        print(f"Available Providers for {selected_modality}: {providers}")
        if providers:
            self.provider_dropdown.configure(values=providers)
            self.provider.set(providers[0])
            print(f"Set default Provider to: {providers[0]}")
        else:
            self.provider_dropdown.configure(values=[])
            self.provider.set('')
            print("No providers available, cleared Provider dropdown.")
        self.update_provider_dropdown()

    def update_provider_dropdown(self, event=None):
        """Update the provider dropdown based on the selected modality and handle cascading model updates."""
        selected_modality = self.modality.get()
        selected_provider = self.provider.get()
        models = self.models_data.get(selected_modality, {}).get(selected_provider, [])
        print(f"Selected Provider: {selected_provider}")
        print(f"Available Models for {selected_provider}: {models}")
        self.model_dropdown.configure(values=models)
        if models:
            self.model_version.set(models[3])
            print(f"Set default Model to: {models[3]}")
        else:
            self.model_version.set('')
            print("No models available, cleared Model dropdown.")
            
    def update_model_dropdown(self, event=None):
        """Handle updates to the model dropdown when the user selects a model."""
        # This function is triggered by the dropdown's command callback when the user makes a selection.
        selected_model = self.model_version.get()  # Get the current selection from the dropdown variable
        print(f"User selected Model: {selected_model}")

    def update_max_tokens_label(self, value):
        """Update the label for max tokens slider."""
        self.max_tokens_value_label.configure(text=str(int(float(value))))

    def update_temperature_label(self, value):
        """Update the label for temperature slider."""
        self.temperature_value_label.configure(text=f"{float(value):0.2f}")

    def save_settings(self):
        """Save the settings and update the variables only when this method is called."""
        # Assuming settings are to be saved to a configuration file or similar
        settings = {
            "modality": self.modality.get(),
            "provider": self.provider.get(),
            "model": self.model_version.get(),
            "max_tokens": int(self.max_tokens_slider.get()),
            "temperature": int(self.temperature_slider.get()),
            "system_prompt": self.system_prompt.get("1.0", "end-1c")
        }
        
        # Update existing instance parameters
        self.llm.update_parameters(
            model_provider=settings["provider"],
            model=settings["model"],
            system_prompt=settings["system_prompt"],
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"]
        )
        
        # Here you can add code to save these settings to a file or apply them wherever necessary
        print("\n\nSettings saved:", settings)
      
    # ---------------------- UI CORE Functions ----------------------
    def enter_press(self, event=None):
        if event.state & 0x0001:  # Shift key is pressed
            self.insert_newline(self)
        else:
            self.send_query()
        
    def insert_newline(self, event=None):
        self.input_field.insert(ctk.END, "\n")
              
    def update_progress_bar(self, duration):
        """Update the progress bar for the specified duration."""
        steps = 100
        step_duration = duration // steps

        def update_step(step=0):
            if step < steps:
                self.progress_bar.after(50, self.progress_bar.set, step / steps * 100)
                # Schedule the next update with a delay
                self.after(step_duration, update_step, step + 1)
            else:
                self.progress_bar.set(100)  # Ensure it reaches 100%

        update_step()

    def update_progress(self, step, max_steps):
        if step <= max_steps:
            self.progress_bar.set(100 * step / max_steps)
            self.after(50, lambda: self.update_progress(step + 1, max_steps))

    def on_close(self):
        """Handle application close by cleaning up resources."""
        self.executor.shutdown(wait=False)
        self.destroy()

    # ---------------------- System Functions ----------------------

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
        for token in self.llm.textGen(user_input, ARV_context):
            print(token, end="", flush=True)
            self.append_conversation(token)
            llm_output += token
        self.append_conversation("\n---\n\n")
        self.last_llm_output = llm_output  # Store last output for image generation
        self.executor.submit(self.tts_output, llm_output)
        
    def tts_output(self, text):
        """Handle text-to-speech functionality."""
        tts_response_path = "output/tmp/tts_llm_response.mp3"
        tts.gen_speech_and_save(text, tts_response_path)
        tts.play_audio(tts_response_path)

    def stt_listen(self):
        """ Continuously listen for transcriptions and put them in the queue. """
        while True:
            transcripted_result = self.stt.run()  # This should be adjusted to non-blocking in `stt.run()`
            if transcripted_result:
                self.transcription_queue.put(transcripted_result)

    def check_transcription(self):
        """ Check the queue for new transcriptions and process them. """
        try:
            transcripted_result = self.transcription_queue.get_nowait()
            print(f">>> TRANSCRIPT: {transcripted_result}\n\n")
            self.append_conversation(f"USER: {transcripted_result}\n\n")
            self.executor.submit(self.process_user_input, transcripted_result)
        except Empty:
            # print(">>> No TRANSCRIPT!!!")
            pass
        finally:
            self.after(500, self.check_transcription)

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
            prompt = self.llm.textGen_promptGen(text)
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
        
def run_app():
    app = ChatGPTUI()
    app.mainloop()
    
if __name__ == "__main__":
    run_app()
