import openai
import os
import pyttsx3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the TTS engine
engine = pyttsx3.init()

def speak(text, rate=200):
    """Speak the given text at the specified rate."""
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

def converse_with_openai(initial_prompt, iterations=10, max_tokens=1800):
    messages = [{'role': 'user', 'content': initial_prompt}]
    roles = [
        'You are an explainer. Build upon the previous message. Answer the question and provide detailed insight. Be concise.', 
        'You are a challenger. Challenge the previous message. Ask a probing question. Be concise.'
    ]

    for i in range(iterations):
        role_instruction = roles[i % 2]

        # Keep only the last 2-3 messages to maintain context without being too verbose
        relevant_messages = messages[-3:]

        # Add the instruction for the model's next response
        relevant_messages.append({'role': 'system', 'content': role_instruction})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=relevant_messages,
            max_tokens=50,  # Limit each agent's response
            temperature=0.7  # Adjust the temperature for variety
        )

        message_content = response.choices[0].message['content'].strip()
        messages.append({'role': 'assistant', 'content': message_content})

        # Print the agent's response with an identifier for clarity
        print(f"A{i+1}: {message_content}")
        
        # Speak the agent's response
        speak(message_content, rate=300)

    return {'messages': messages}

# Get theme from user
theme = input("Enter the theme of the conversation: ")
conversation_log = converse_with_openai(theme, iterations=5)
