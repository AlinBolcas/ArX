import time
import random
import math

import os
from dotenv import load_dotenv
import threading

# Load the .env file
load_dotenv()

# Read the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")


from openai import OpenAI
client = OpenAI()


chat_history = {"content": ""}
# Variable to store chat history

# Global variable to track the last prompt time
last_prompt_time = time.time()


# Placeholder for LLM interaction
def llm_response(prompt, chat_history):

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are humanity, art, love, life, time, gods, you are everything."},
            {"role": "user", "content": "Given this past conversation:"
             f"{chat_history}"
             f"continue from this latest message: {prompt}"},
        ],
        temperature=0.5,
        max_tokens=150,
    )
    return response.choices[0].message.content

# Placeholder for long-term memory retrieval
def retrieve_long_term_memory(chat_history):
    """
    Summarize the chat history if it exceeds a certain length, maintaining elements
    that may require asking a follow-up question.
    """
    # Set a threshold for the length of chat history (e.g., 1000 words)
    WORD_COUNT_THRESHOLD = 500
    
    conversation_text = chat_history["content"]

    # Count the number of words in the chat history
    word_count = len(conversation_text.split())
    
    history = ""
    
    if len(conversation_text.split()) > WORD_COUNT_THRESHOLD:
        # Format the chat history for summarization request
        # formatted_history = '\n'.join(f"{line}" for line in chat_history.split('\n'))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at summarising conversations."},
                {"role": "user", "content": f"Summarise this conversation {conversation_text} while maintaining the format with 'User:' and 'AI:' as a chat."
                 "Please Ensure that the summarized content is still meaningful and relevant to the ongoing conversation."}
            ],
            temperature=0.5,
        )
        history = response.choices[0].message.content
    else:
        history = conversation_text

    # Write the (summarized or original) history to a markdown file
    with open("chat_history.txt", "a") as file:
        file.write(history + "\n\n")
    # print("[Debug] Chat history saved to 'chat_history.txt'")
    return history

# Placeholder for fine-tuned LLM for question asking
def fine_tuned_llm_for_question_asking(context):
    # In a real implementation, this would be a specialized model fine-tuned for question generation.
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are an expert at asking single simple conversation stimulating, soul searching, empathic question."},
            {"role": "user", "content": 
            f"Ask a question given the context: {context}"},
        ],
        temperature=0.8,
        max_tokens=50,
    )
    return response.choices[0].message.content

def double_pendulum_trigger():
    # Rename the chaotic trigger function for consistency
    t = time.time() % 30  
    return abs(math.sin(t))

def determine_contextual_need(context):
    # Placeholder for assessing the contextual need for a question
    # Implement logic to determine if there's a need for a question based on chat history
    """
    Assess the contextual need for a question based on the provided context.
    Returns a float value between 0 and 1 indicating the likelihood of needing to ask a question.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": 
                "You are an expert at determining if a question must be asked given a context."},
                {"role": "user", "content": 
                "Return a single high-float value from 0 to 1 indicating " 
                f"the likelihood that a question could be asked based on this context: {context}"},
            ],
            temperature=1.2,
            max_tokens=20,
        )
        urgency_score = float(response.choices[0].message.content)
    except ValueError:
        # Handle cases where the response is not a valid float
        urgency_score = 0.5
    except Exception as e:
        # Handle any other exceptions that might occur
        print(f"urgency score: {e}")
        urgency_score = 0.5
    # Ensure the score is within the 0 to 1 range
    return max(0, min(1, urgency_score))

def calculate_score(time_elapsed, contextual_need, chaotic_factor):
    time_factor = min(time_elapsed / 30, 1)  # Normalized to 0-1
    need_factor = min(contextual_need * 3, 3)   # Stronger need increases the factor

    # Chaos factor could be used to occasionally boost the score
    # chaos_boost = chaotic_factor * random.uniform(0.5, 1.5)

    score = time_factor * need_factor * chaotic_factor
    
    # print(f"[Debug] Time Factor: {time_factor}, Need Factor: {need_factor}, Chaos Boost: {chaos_boost}, Score: {score}")
    return max(0, min(1, score))

def update_chat_history(user_prompt, response, chat_history):
    current_time = time.time()
    chat_history["content"] += f"User: {user_prompt}\n"  # Update chat history
    chat_history["content"] += f"AI: {response}\n"  # Update chat history
    return chat_history, current_time

def spontaneous_questioning(chat_history):
    global last_prompt_time
    TRIGGER_THRESHOLD = 0.7

    while True:
        time_elapsed = time.time() - last_prompt_time
        context = chat_history["content"]
        contextual_need = determine_contextual_need(chat_history["content"])
        trigger_score = calculate_score(time_elapsed, contextual_need, double_pendulum_trigger())

        # print(f"[Debug] Time: {time_elapsed}, Score: {trigger_score}, Context Need: {contextual_need}")
        
        if trigger_score > TRIGGER_THRESHOLD:
            question = fine_tuned_llm_for_question_asking(context)
            print("\nAI Question:", question)
            chat_history["content"] += f"AIQ: {question}\n"
            # print(f"[Debug] Asking Question: {question}")
            time.sleep(5)  # Add a delay before the next potential question

        time.sleep(1)  # Check every 1 second

def main():
    global last_prompt_time
    global chat_history
    
    
    with open("chat_history.txt", "w") as file:
        file.write("")
    
    # Start the spontaneous questioning in a separate thread
    threading.Thread(target=spontaneous_questioning, args=(chat_history,)).start()


    while True:
        user_prompt = input("User: ")
        last_prompt_time = time.time()
        
        response = llm_response(user_prompt, chat_history["content"])

        chat_history, _ = update_chat_history(user_prompt, response, chat_history)

        # Summarize the chat history after each user input
        chat_history["content"] = retrieve_long_term_memory(chat_history)

        print("AI Response:", response)

        if user_prompt.lower() == 'exit':
            break

if __name__ == "__main__":
    main()

