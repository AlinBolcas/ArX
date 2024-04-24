from multiprocessing import Process, Queue
from queue import Empty
import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize task queue and data store
task_queue = Queue()

# Initialize agents with specific roles and prompts
agents = {
    'philosopher': 'Provide ethical and philosophical perspectives on {}.',
    'scientist': 'Offer empirical evidence and scientific viewpoints on {}.',
    'artist': 'Consider the creative and aesthetic aspects of {}.',
    'historian': 'Provide historical context and lessons regarding {}.',
}

def specialized_agent(q, query, agent_role):
    agent_prompt = agents[agent_role].format(query)

    message = [{"role": "system", "content": agent_role},
               {"role": "user", "content": query}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=1.2,
        max_tokens=330,
        frequency_penalty=0.0
    )
    output = response['choices'][0]['message']['content'].strip()
    q.put({agent_role: output})

def arbitrator_agent(q, data_store):
    perspectives = '\n'.join([f"{k.capitalize()}: {v}" for k, v in data_store.items()])
    prompt = "Synthesize these perspectives."
    
    message = [{"role": "system", "content": "You are the jury in a court of debates. Make the final decision for the output of an ansamble of AI agents."},
               {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=1.2,
        max_tokens=1000,
        frequency_penalty=0.0
    )
    output = response['choices'][0]['message']['content'].strip()
    q.put({'arbitrator': output})

if __name__ == "__main__":
    query = "the implications of AI in society"
    
    # Query each specialized agent
    processes = [Process(target=specialized_agent, args=(task_queue, query, role)) for role in agents.keys()]
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
        
    data_store = {}
    while not task_queue.empty():
        try:
            result = task_queue.get_nowait()
            data_store.update(result)
        except Empty:
            break

    # Arbitrate among the agents
    arbitrator_process = Process(target=arbitrator_agent, args=(task_queue, data_store))
    arbitrator_process.start()
    arbitrator_process.join()
    
    data_store.update(task_queue.get())
    
    # Print the perspectives and their synthesis
    print("Results from specialized agents:")
    for key, value in data_store.items():
        print(f"{key.capitalize()}:\n{value}\n")
