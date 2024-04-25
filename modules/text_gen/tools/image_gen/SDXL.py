import os, json
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image
import torch


# Global variables to hold the loaded models
text_pipe = None
image_pipe = None

def prepare_image(image):
    # check if 1k or larger -> crop to 1k
    # check if 512 -> crop to 512
    width, height = image.size
    new_width, new_height = size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))

def init_models():
    model_thread = threading.Thread(target=load_models)
    model_thread.start()

def load_models():
    global text_pipe, image_pipe
    if text_pipe is None:
        text_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        text_pipe.to("mps")
    
    if image_pipe is None:
        image_pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        image_pipe.to("mps")

def SDXL_Turbo(prompt, image=None, steps=2, strength=0.7, guidance=0.5):
    try:
        print("Generating SDXL_Turbo image...")
        if image is not None:
            image = prepare_image(image)
            # Use image_pipe for image-to-image generation
            generated_image = image_pipe(prompt=prompt, image=init_image, num_inference_steps=steps, strength=strength, guidance_scale=guidance).images[0]
        else:
            # Use text_pipe for text-to-image generation
            generated_image = text_pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
        
        generated_image = generated_image.resize((1024, 1024))
        return generated_image
    except Exception as e:
        print(f"2DGen, SDXL_Turbo ERROR: {e}")
        return None

def SDXL_Turbo_Batch(prompt, image=None):
    # stylish, creative, focused, unique
    
    # Define lists for the parameters to iterate through
    strength_values = [0.9, 0.7, 0.5]
    num_inference_steps_values = [1, 2, 3]
    guidance_scale_values = [0, 0.5, 1]

    # Iterate through the parameters and generate comparison images
    for strength_value in strength_values:
        for num_steps in num_inference_steps_values:
            for scale_value in guidance_scale_values:
                # Generate the image using the constant prompt and varying parameters
                if (num_steps*strength_value) >= 1:
                    return SDXL_Turbo(prompt, image=None, steps=2, strength=0.7, guidance=0.5)


if __name__ == "__main__":
    main()