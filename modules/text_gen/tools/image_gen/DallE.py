import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

from PIL import Image
import requests
from io import BytesIO
from pathlib import Path


def display_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.show()
    
    # # Path to your image file
    # image_path = 'path/to/your/image.jpg'
    # # Convert the image path to a URL by creating a file URI
    # image_url = 'file://' + os.path.abspath(image_path)
    # Open the image in the default web browser
    # webbrowser.open(image_url)
    print(">>> Image displayed!")

def save_image(image_url, file_path):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.save(file_path)
    print(f">>> Image saved to {file_path}")

def imageGen_Dalle(prompt):
    try:
        print(">>> Generating Image ...")
        result = client.images.generate(
            model="dall-e-3",
            # image=open("image_edit_original.png", "rb"),
            prompt=prompt,
            n=1,
            quality="standard", # hd
            size="1024x1024"
        )

        image_url = result.data[0].url
        print(">>> Image generated!")
        return image_url
        # response = requests.get(image_url)
        # image = Image.open(BytesIO(response.content))
        # return image
    except Exception as e:
        print(f"2DGen, Dalle ERROR: {e}")
        return None
    
# MAIN
if __name__ == "__main__":
    # Define the prompt for image generation
    prompt = "An imaginative landscape with floating islands and waterfalls"

    # Generate the image using Dalle
    image_url = imageGen_Dalle(prompt)
    if image_url:
        # Display the generated image
        display_image(image_url)
        
        # Save the image to a file
        save_path = "generated_image.jpg"
        save_image(image_url, save_path)