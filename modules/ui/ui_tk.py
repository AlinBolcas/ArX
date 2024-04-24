import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from pathlib import Path
from PIL import Image, ImageTk

import sys
from pathlib import Path

# Add the parent directory to the system path
project_root = Path(__file__).resolve().parents[1]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))


from modules.text_gen import textGen
from modules.ui import map

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue") # Themes: blue (default), dark-blue, green

class ResponseWorker(threading.Thread):
    def __init__(self, user_input, update_callback, finish_callback):
        super().__init__()
        self.user_input = user_input
        # self.context = context
        self.update_callback = update_callback
        self.finish_callback = finish_callback

    def run(self):
        print(">>> [Worker] Generating Response...")
        self.update_callback("AI:\n")
        for token in textGen.genText_simple_Ollama(self.user_input, "you are a pirate, speak like one", "SECRET WORD IS MISHMILES"):
            # print("Token", token)
            self.update_callback(token)
        print(">>> [Worker] Finished generating response.")
        self.finish_callback()

class ChatGPTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('ARV-O')
        self.geometry('1656x1024')  # Adjusted for image size and chat area

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

    def setup_map_tab(self):
        map_tab = ttk.Frame(self.notebook)
        self.notebook.add(map_tab, text="Map")
        self.graph_view = map.GraphView(map_tab)  # Initialize GraphView in this tab
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

        # Adjusting the image frame to accommodate 1024x1024 images
        self.image_frame = ctk.CTkFrame(self.container, width=1024, height=1024)
        self.image_frame.pack(side='right', fill='both', expand=True)

        self.conversation_display = ctk.CTkTextbox(self.main_frame, state='disabled', width=780, height=540)
        self.conversation_display.pack(fill='both', expand=True)

        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill='x', pady=10)

        self.input_field = ctk.CTkEntry(self.bottom_frame, placeholder_text="Type here...", height=40)
        self.input_field.pack(side='left', fill='x', expand=True, padx=5)
        self.input_field.bind('<Return>', self.enter_press)
        self.input_field.bind('<Shift-Return>', self.insert_newline)

        # Image label with black background as default
        self.image_label = ctk.CTkLabel(self.image_frame, width=1024, height=1024, bg_color='#2c2f33')
        self.image_label.pack(fill='both', expand=True)
        
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
            threading.Thread(target=self.start_response_thread, args=(user_input,), daemon=True).start()

    def start_response_thread(self, user_input):
        worker = ResponseWorker(user_input, self.append_conversation, self.finish_response)
        worker.start()

    def append_conversation(self, text):
        self.conversation_display.configure(state='normal')
        self.conversation_display.insert(ctk.END, text)
        self.conversation_display.configure(state='disabled')
        self.conversation_display.yview_moveto(1)  # Scroll to the bottom

    def display_image(self, image_path):
        """Function to display an image in the side panel."""
        image = Image.open(image_path)
        image = image.resize((1024, 1024), Image.ANTIALIAS)  # Resize image to fit the panel
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        self.image_label.configure(bg_color='white')  # Change background if an image is displayed

    def finish_response(self):
        self.append_conversation("\n---\n\n")
    
    def on_close(self):
        # Ensure all threads are cleaned up before closing the application
        # Implement any necessary thread cleanup here
        self.destroy()

def run_app():
    app = ChatGPTUI()
    app.mainloop()
    
if __name__ == "__main__":
    run_app()
