import os
import requests
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import filedialog, messagebox
from google_images_download import google_images_download


load_dotenv()

openai.api_key = os.getenv('openAIKey')


def scrape_google_images(query, num_images):
    response = google_images_download.googleimagesdownload()
    arguments = {
        "keywords": query,
        "limit": num_images,
        "print_urls": True,
        "size": ">1024*768",  # Only download images larger than 1024x768 pixels
        "no_directory": True  # Don't create a directory for the images
    }
    paths = response.download(arguments)
    img_urls = paths[0][query]
    return img_urls

def expand_theme_gpt(theme, instruction, num_branches):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a creative character artist assistant. I want you to provide a list of related keywords or topics for image collection. "
                "Ensure they are contextually related to the main theme or subject for Google searching purposes. "
                f"For example, if the theme is 'dragons', some related topics could be: "
                "'dragon anatomy', 'dragon concept art', 'dragon sketches', 'dragon scales', 'dragon drawings', 'dragon designs', and so on, but use what's most relevant to the theme instead of these examples exactly. "
                f"Specific Requirements: {instruction}. "
                f"Given these requirements, please provide ideal google search syntaxes for the theme provided. If instructions is '-', you're not constrained by them."
            )
        },
        {"role": "user", "content": f"Given the theme '{theme}', what are the essential related google search topics needed to inform the creation of a CGI project on the given theme?"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.3,
    )
    
    expanded_themes = [item.strip() for item in response.choices[0].message['content'].split('\n') if item][:num_branches]
    print(f"Expanded themes for '{theme}': {expanded_themes}")
    return expanded_themes

def sanitize_filename(filename):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename

def explore_images():
    theme = theme_entry.get()
    instruction = instruction_entry.get()
    num_branches = int(num_branches_entry.get())
    num_images = int(num_images_entry.get())
    file_location = filedialog.askdirectory(title="Select a Folder")

    theme_folder = os.path.join(file_location, theme)
    if not os.path.exists(theme_folder):
        os.makedirs(theme_folder)

    expanded_themes = [theme] + expand_theme_gpt(theme, instruction, num_branches - 1)
    for branch in expanded_themes:
        img_paths = scrape_google_images(f"{theme} {branch}", num_images)
        for img_path in img_paths:
            print(f"Saved image to: {img_path}")
            # Move the image to the desired directory
            new_path = os.path.join(theme_folder, os.path.basename(img_path))
            os.rename(img_path, new_path)

    messagebox.showinfo("Info", "Downloading complete!")

app = tk.Tk()
app.title("Image Scraper")

theme_label = tk.Label(app, text="Theme:")
theme_label.pack(pady=10)
theme_entry = tk.Entry(app)
theme_entry.pack(pady=10)

instruction_label = tk.Label(app, text="Instruction:")
instruction_label.pack(pady=10)
instruction_entry = tk.Entry(app)
instruction_entry.pack(pady=10)

num_branches_label = tk.Label(app, text="Number of Search Branches:")
num_branches_label.pack(pady=10)
num_branches_entry = tk.Entry(app)
num_branches_entry.pack(pady=10)

num_images_label = tk.Label(app, text="Number of Images per Search Branch:")
num_images_label.pack(pady=10)
num_images_entry = tk.Entry(app)
num_images_entry.pack(pady=10)

explore_button = tk.Button(app, text="Explore", command=explore_images)
explore_button.pack(pady=20)

app.mainloop()
