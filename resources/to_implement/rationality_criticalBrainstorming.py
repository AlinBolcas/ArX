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

def summarize_conversation(messages):
    third = len(messages) // 3
    oldest_messages = messages[:third]
    middle_messages = messages[third:2*third]
    recent_messages = messages[2*third:]
    
    oldest_text = ' '.join([msg['content'] for msg in oldest_messages])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': f"Summarize the following conversation: {oldest_text}"}]
    )
    summary = response.choices[0].message['content'].strip()

    summarized_messages = [{'role': 'user', 'content': summary}] + middle_messages + recent_messages
    return summarized_messages

def converse_with_openai(initial_prompt, iterations=10, max_tokens=1800):
    messages = [{'role': 'user', 'content': initial_prompt}]
    roles = ['You bring an out-of-the-box thought-provoking idea to the topic of discussion. Answer any raised question and ask a followup question. Be concise, precise, and profound. Keep your whole reply within 2-3 sentences or 35-65 words.', 
             'You challenge any raised topic of discussion. Answer the question and ask a thoughtful followup question. Be concise, precise, and profound. Keep your whole reply within 2-3 sentences or 35-65 words.']

    for _ in range(iterations):
        role_instruction = roles[_ % 2]

        # Add the instruction for the model's next response
        messages.append({'role': 'system', 'content': role_instruction})

        # Adjusting the temperature for more randomness
        temp_value = 0.7 + (_ % 2) * 0.1  # This will alternate between 0.7 and 0.8 for variety

        # Only send the last few messages to keep it relevant and avoid reaching token limits
        recent_messages = messages[-3:]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=recent_messages,
            max_tokens=65,  # Slightly increase the token limit
            temperature=temp_value  # Adjust the temperature
        )

        message_content = response.choices[0].message['content'].strip()
        messages.append({'role': 'assistant', 'content': message_content})

        # Print the agent's response
        print(f"A{_ + 1}: {message_content}")
        
        # Speak the agent's response
        speak(message_content, rate=500)

        # Check if the conversation exceeds the token limit
        if sum([len(msg['content']) for msg in messages]) > max_tokens:
            messages = summarize_conversation(messages)

    return {'messages': messages}



# Get theme from user
theme = input("Enter the theme of the conversation: ")
conversation_log = converse_with_openai(theme, iterations=5)
