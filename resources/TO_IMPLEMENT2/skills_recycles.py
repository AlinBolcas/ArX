import threading
import os
import json
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI


# Load the .env file
load_dotenv()

# Read the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# ---------------------------------------------------------
# chat completions
# ---------------------------------------------------------

def CO_thought_stream(prompt):
    try:
        print("Generating text...")
        # An example of a system message that primes the assistant to give brief, to-the-point answers
        completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are humanity, art, love, life, time, gods, you are everything."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.9,
        stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Error in generating text: {e}")
        return None

def mem_thought_stream(prompt, chat_history):
    try:
        print("Generating text...")
        # Combine active memory (formatted) and long-term memory
        messages = chat_history["active_memory"]
        if chat_history["long_term_memory"]:
            messages.append({"role": "system", "content": f"You always reply in short concise expert-master level sentences. "
                                                        f"Conversation background information: \n{chat_history['long_term_memory']}"})

        # Append the latest prompt to the messages
        messages.append({"role": "user", "content": prompt})

        completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.618,
        max_tokens=350,
        stream=True
        )

        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"Error in generating text: {e}")
        return None

# Note: Placeholder LLM Response Function
def simple_thought(prompt, chat_history):
    # Combine active memory (formatted) and long-term memory
    messages = chat_history["active_memory"]
    
    messages.append({"role": "system", "content": f"You are an expert in taking long answers and reducing them to the most bare minimum essentials meant for speech generation."})

    # Append the latest prompt to the messages
    messages.append({"role": "user", "content": f"Rephrase this response in the shortest format possible: {prompt}"})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0.618,
        max_tokens=150,
    )
    return response.choices[0].message.content

def librarian(active_memory, system_history):

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": f"You are an expert at summarising conversations within a given context: '\n{system_history}'"},
            {"role": "user", "content": "Summarise this exchange to the most important elements to remember:\n"
             f"'{active_memory}'"},
        ],
        temperature=0.618,
        # max_tokens=150,
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# Images
# ---------------------------------------------------------

def visual_perception(image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in a sentence."},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    }
                ],
            }
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content

def image_prompt(context):
    file_path = Path(__file__).parent / 'data'
    with open(str(file_path / 'image_prompts.json'), 'r') as file:
        image_prompts_data = json.load(file)

    # Read Markdown file (assuming it contains additional context or template)
    with open(str(file_path / 'image_sys_prompt.md'), 'r') as md_file:
        instructions = md_file.read()

    # Prepare messages for GPT-4 Completion
    messages = [
        {"role": "system", "content": f"You are art. Use these instructions to generate image prompts: \n {instructions}"},
    ]

    # Adding prompts from JSON data
    for prompt in image_prompts_data:
        messages.append({"role": "user", "content": prompt["user"]})
        messages.append({"role": "assistant", "content": prompt["assistant"]})

    # Optionally, add content from markdown file
    messages.append({"role": "user", "content": f"Generate an image prompt in the same format as before, but representative of this message: \n{context}"})

    print("Generating image prompt(s)...")
    
    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=messages,
    max_tokens=150,
    temperature=1.2,
    )
    return response.choices[0].message.content

def generate_image(prompt):
    try:
        print("Generating image...")
        result = client.images.generate(
            model="dall-e-3",
            # image=open("image_edit_original.png", "rb"),
            prompt=prompt,
            n=1,
            quality="standard", # hd
            size="1024x1024"
        )

        image_url = result.data[0].url
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error in generating image: {e}")
        return None

# def generate_quick_image(prompt):
#     try:
#         print("Generating image...")
#         result = client.images.generate(
#             model="dall-e-2",
#             # image=open("image_edit_original.png", "rb"),
#             prompt=prompt,
#             n=1,
#             # quality="hd",
#             size="512x512"
#         )

#         image_url = result.data[0].url
#         response = requests.get(image_url)
#         image = Image.open(BytesIO(response.content))
#         return image
#     except Exception as e:
#         print(f"Error in generating image: {e}")
#         return None

# ---------------------------------------------------------
# Audio
# ---------------------------------------------------------

def transcribe_speech(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text

def generate_speech(text, file_path):
    # Generate the speech file
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",  # alloy let it choose voice & intonation based on context input
        input=text,
        speed=1.35,
    )
    # Save the audio file to the specified path
    response.stream_to_file(file_path)
    return file_path


# ---------------------------------------------------------
# main for debugging
# ---------------------------------------------------------

def main():
    output_dir = Path(__file__).parent / 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Paths for image and speech files
    image_path = output_dir / "generated_image.png"
    speech_file_path = output_dir / "speech.mp3"

    chat_history = ""  # Variable to store chat history
    system_history = ""  # Variable to store system messages

    prompt = f"{chat_history}User: " + input("Enter your text prompt: ")
    final_message = ""
    print("Generated text:", end="")
    for message in generate_text(prompt):
        print(message, end="")
        final_message += message
    chat_history += f"\nAI: {final_message}\n\n"

    # Start the image generation in a separate thread
    image_thread = threading.Thread(target=lambda: save_image(final_message, str(image_path)))
    image_thread.start()

    # Start the speech generation in a separate thread
    speech_thread = threading.Thread(target=lambda: save_speech(final_message, str(speech_file_path)))
    speech_thread.start()

    # Wait for threads to complete (optional, remove if you don't need to wait)
    image_thread.join()
    speech_thread.join()

if __name__ == "__main__":
    main()