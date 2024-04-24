import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
from time import sleep

# Dummy functions for demonstration.
def dummy_function(*args):
    print("Function called with arguments:", args)

class ChatGPTUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Comprehensive Demo')
        self.geometry('1200x800')

        # Container for the tabs
        notebook_container = tk.Frame(self)
        notebook_container.pack(fill='both', expand=True)
        self.notebook = ttk.Notebook(notebook_container)
        self.notebook.pack(fill='both', expand=True)

        # Create CTkTabView
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(fill='both', expand=True, padx=10, pady=10)

        # Setup tabs
        self.setup_chat_tab2()
        self.setup_widgets_demo_tab2()
        # More tabs can be setup similarly

        # Setup tabs
        self.setup_chat_tab()
        self.setup_widgets_demo_tab()
        self.setup_controls_tab()
        self.setup_visuals_tab()
        self.setup_misc_tab()

    def setup_chat_tab(self):
        chat_tab = ctk.CTkFrame(self.notebook)
        ctk.CTkLabel(chat_tab, text="Chat interface not implemented").pack(pady=10)
        self.notebook.add(chat_tab, text='Chat')

    def setup_widgets_demo_tab(self):
        widgets_tab = ctk.CTkFrame(self.notebook)
        self.setup_widgets_demo(widgets_tab)
        self.notebook.add(widgets_tab, text='Widgets Demo')

    def setup_widgets_demo(self, container):
        # Radio Button
        radio_var = tk.IntVar()
        ctk.CTkRadioButton(container, text="Option 1", variable=radio_var, value=1, command=lambda: dummy_function("Radio 1")).pack(pady=10)
        ctk.CTkRadioButton(container, text="Option 2", variable=radio_var, value=2, command=lambda: dummy_function("Radio 2")).pack(pady=10)

        # Slider
        slider = ctk.CTkSlider(container, from_=0, to=100, command=lambda value: dummy_function("Slider", value))
        slider.pack(pady=20)

        # Correctly instantiating a CTkProgressBar with a master widget and setting its value.
        progress = ctk.CTkProgressBar(container, width=200)  # 'container' is the parent widget
        progress.pack(pady=30)
        progress.set(0.5)  # Correctly setting the progress bar's value to 50%

        # Toggle
        toggle = ctk.CTkSwitch(container, text="Toggle", command=lambda: dummy_function("Toggle"))
        toggle.pack(pady=40)

        # Dropdown
        option_var = tk.StringVar()
        options = ["Option 1", "Option 2", "Option 3"]
        dropdown = ctk.CTkOptionMenu(container, variable=option_var, values=options, command=dummy_function)
        dropdown.pack(pady=50)

    def setup_controls_tab(self):
        controls_tab = ctk.CTkFrame(self.notebook)
        self.setup_controls(controls_tab)
        self.notebook.add(controls_tab, text='Controls')

    def setup_controls(self, container):
        # Button
        ctk.CTkButton(container, text="Click Me", command=lambda: dummy_function("Button Clicked")).pack(pady=10)

    def setup_visuals_tab(self):
        visuals_tab = ctk.CTkFrame(self.notebook)
        self.setup_visuals(visuals_tab)
        self.notebook.add(visuals_tab, text='Visuals')

    def setup_visuals(self, container):
        # Label
        ctk.CTkLabel(container, text="This is a label").pack(pady=10)
        # Entry
        ctk.CTkEntry(container, placeholder_text="This is an entry").pack(pady=10)

    def setup_misc_tab(self):
        misc_tab = ctk.CTkFrame(self.notebook)
        self.setup_misc(misc_tab)
        self.notebook.add(misc_tab, text='Miscellaneous')

    def setup_misc(self, container):
        # CheckBox
        checkbox_var = tk.IntVar()
        ctk.CTkCheckBox(container, text="Check me", variable=checkbox_var, onvalue=1, offvalue=0, command=lambda: dummy_function("Checkbox")).pack(pady=10)
        



    def setup_chat_tab2(self):
        chat_tab = ctk.CTkFrame(self.tab_view)
        self.tab_view.add(chat_tab)
        # Setup chat interface in chat_tab

    def setup_widgets_demo_tab2(self):
        widgets_tab = ctk.CTkFrame(self.tab_view)
        self.setup_widgets_demo2(widgets_tab)
        self.tab_view.add(widgets_tab)

    def setup_widgets_demo2(self, container):
        # Setup different widgets for demonstration
        progress = ctk.CTkProgressBar(container, width=200, height=20)
        progress.pack(pady=20)
        # Set value using the set method
        progress.set(0.5)


def run_app():
    app = ChatGPTUI()
    app.mainloop()

if __name__ == "__main__":
    run_app()
