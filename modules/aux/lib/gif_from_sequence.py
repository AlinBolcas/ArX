import sys
from pathlib import Path
import customtkinter as ctk
import imageio
from PIL import Image, ImageSequence
import os
from tkinter import filedialog, messagebox
import json  # Import json for settings handling


def gifGen(input_paths, output_path, duration=0.18):
    """Generate a GIF from a list of image file paths using Pillow for improved quality."""
    try:
        # Load images
        frames = [Image.open(path) for path in input_paths]
        
        # Optimize images by converting them to a palette-based format with an adaptive palette
        frames = [frame.convert('P', palette=Image.ADAPTIVE, colors=256, dither=Image.FLOYDSTEINBERG) for frame in frames]

        # Save the frames as a GIF
        frames[0].save(
            output_path, 
            save_all=True,  # True or False
            append_images=frames[1:], 
            optimize=True,  # 
            duration=int(duration * 1000), 
            loop=0,
            dither=Image.FLOYDSTEINBERG # FLOYDSTEINBERG or NONE
        )
        print("Success: The GIF has been created successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        

class GIFGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GIF Generator")
        self.geometry("600x250")  # Adjusted for additional inputs

        # # Define the path for settings.json
        # self.settings_file = os.path.join(Path.home(), "settings.json")
        # print(f"Settings file path: {self.settings_file}")  # Print the path where settings will be saved and loaded

        # Define the path for settings.json based on the script's location
        self.settings_file = os.path.join(Path(__file__).parent, "path_settings.json")
        print(f"Settings file path: {self.settings_file}")  # Print the path where settings will be saved and loaded

        self.load_settings()  # Load settings if available

        # Input path setup
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=20, padx=20, fill='x')
        self.input_label = ctk.CTkLabel(self.input_frame, text="Select Image Files:")
        self.input_label.pack(side="left", padx=10)
        self.input_entry = ctk.CTkEntry(self.input_frame, width=120, placeholder_text="Select the folder with images...")
        self.input_entry.pack(side="left", fill="x", expand=True)
        self.input_button = ctk.CTkButton(self.input_frame, text="Browse", command=self.select_input_files)
        self.input_button.pack(side="right", padx=10)

        # Output path setup
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.pack(pady=10, padx=20, fill='x')
        self.output_label = ctk.CTkLabel(self.output_frame, text="Output GIF Path:")
        self.output_label.pack(side="left", padx=10)
        self.output_entry = ctk.CTkEntry(self.output_frame, width=120, placeholder_text="Specify the output GIF file path...")
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.output_button = ctk.CTkButton(self.output_frame, text="Save As", command=self.select_output_file)
        self.output_button.pack(side="right", padx=10)

        # Duration entry
        self.duration_frame = ctk.CTkFrame(self)
        self.duration_frame.pack(pady=10, padx=20, fill='x')
        self.duration_label = ctk.CTkLabel(self.duration_frame, text="Frame Duration (seconds):")
        self.duration_label.pack(side="left", padx=10)
        self.duration_entry = ctk.CTkEntry(self.duration_frame, width=120, placeholder_text="Enter frame duration, e.g., 0.1")
        self.duration_entry.pack(side="left", fill="x", expand=True)

        # Generate button
        self.generate_button = ctk.CTkButton(self, text="Generate GIF", command=self.start_gif_creation)
        self.generate_button.pack(pady=20)

    def load_settings(self):
        try:
            with open(self.settings_file, "r") as f:
                self.settings = json.load(f)
            print("Settings loaded successfully.")
        except FileNotFoundError:
            self.settings = {}
            print("No settings file found. A new one will be created.")

    def save_settings(self):
        with open(self.settings_file, "w") as f:
            json.dump(self.settings, f)
        print(f"Settings saved to {self.settings_file}")

    def select_input_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_paths:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, "; ".join(file_paths))
            self.settings['last_input_path'] = "; ".join(file_paths)
            self.save_settings()

    def select_output_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".gif")
        if file_path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, file_path)
            self.settings['last_output_path'] = file_path
            self.save_settings()

    def start_gif_creation(self):
        input_paths = self.input_entry.get().split("; ")
        output_path = self.output_entry.get()
        try:
            duration = float(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid duration input. Please enter a numeric value.")
            return

        if not input_paths or not output_path:
            messagebox.showerror("Error", "Please specify both input and output paths.")
            return

        try:
            gifGen(input_paths, output_path, duration)
            messagebox.showinfo("Success", "The GIF has been created successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = GIFGeneratorApp()
    app.mainloop()
