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


from modules.text_gen import textGen as tg
from modules.ui import map

ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue") # Themes: blue (default), dark-blue, green



# TEMP
arvolve_context = """
Arvolve stands at the forefront of blending artistic creativity with cutting-edge technology, pioneering the evolution of art through advanced 3D CGI and VFX productions. Founded by Alin Bolcas, a visionary artist and technologist, the company excels in character concept design, crafting mesmerizing visuals that captivate global audiences. Arvolve's commitment to innovation extends into the realm of artificial intelligence, where it develops ARV-O, a multimodal AI system inspired by human cognition. This system enhances artistic workflows, facilitates creative ideation, and fosters a deeper exploration of AI's potential in arts and consciousness. With a robust portfolio of high-profile projects and a dynamic approach to AI and CGI integration, Arvolve is dedicated to redefining creative expression and advancing humanity through its pioneering work.
""" 

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
        
        prompt = f"Reply solely with a list of 3 relevant elements and nothing else, no introductory sentance. The list should have a new line for each element and responses to user input: {self.user_input}"
        
        print(prompt, "\n")
        response_output = ""
        for token in tg.textGen(user=prompt,
                            system="You are an expert assistant at making comprehansive lists to expand on topics and ideas.",    
                            provider="Ollama"):
            print(token, end="", flush=True)
            # self.update_callback(token)
            response_output += token
            
        # Splitting the accumulated response into lines and stripping whitespace
        llm_list = [token.strip() for token in response_output.split('\n') if token.strip()]
        
        print("\n", "REPLYING TO LIST OF REPLIES", "\n")
        for prompt in llm_list:
            
            print("\n", prompt, "\n")
            self.update_callback(f"\n\n EXPANDING on: {prompt}\n")
            
            for token in tg.textGen(prompt, "You are a world class expert in all fields and disciplines of the world. You intelligently take long answers and reduce them to the most bare minimum essentials. You are ARV-O, a creative AI assistant. You are an employee at Arvolve. This is your company's ethos: {context}",
                                    arvolve_context, provider="Ollama"):
                print(token, end="", flush=True)

                self.update_callback(token)

            tg.textGen_tools_agent(prompt)

        print(">>> [Worker] Finished generating response.")
        self.finish_callback()

class ChatGPTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('ARV-O')
        # self.geometry('1656x1024')  # Adjusted for image size and chat area
        self.geometry('600x800')  # Adjusted for image size and chat area
        
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

        # # Adjusting the image frame to accommodate 1024x1024 images
        # self.image_frame = ctk.CTkFrame(self.container, width=1024, height=1024)
        # self.image_frame.pack(side='right', fill='both', expand=True)

        # # Image label with black background as default
        # self.image_label = ctk.CTkLabel(self.image_frame, width=1024, height=1024, bg_color='#2c2f33')
        # self.image_label.pack(fill='both', expand=True)

        self.conversation_display = ctk.CTkTextbox(self.main_frame, state='disabled', width=780, height=540)
        self.conversation_display.pack(fill='both', expand=True)

        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill='x', pady=10)

        self.input_field = ctk.CTkEntry(self.bottom_frame, placeholder_text="Type here...", height=40)
        self.input_field.pack(side='left', fill='x', expand=True, padx=5)
        self.input_field.bind('<Return>', self.enter_press)
        self.input_field.bind('<Shift-Return>', self.insert_newline)


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
