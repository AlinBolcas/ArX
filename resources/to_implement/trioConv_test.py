from multiprocessing import Process, Queue
from queue import Empty
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')


def specialized_agent(q, query, sector):
    prompt = f"{query} from a {sector} perspective"
    message = [{"role": "system", "content": sector},
               {"role": "user", "content": query}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=0.2,
        max_tokens=200,
        frequency_penalty=0.0
    )
    output = response['choices'][0]['message']['content'].strip()
    q.put({'sector': sector, 'response': output})

def arbitrator_agent(q, cognitivism, behaviorism):
    context = f"Cognitivism: {cognitivism}\nBehaviorism: {behaviorism}"
    prompt = "Synthesize these perspectives."
    message = [{"role": "system", "content": "arbitrator"},
               {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        temperature=0.2,
        max_tokens=200,
        frequency_penalty=0.0
    )
    output = response['choices'][0]['message']['content'].strip()
    q.put({'sector': 'arbitrator', 'response': output})

if __name__ == '__main__':
    perspectives = ["cognitivism", "behaviorism"]
    base_query = "Exercise vs Sedentary on lifespawn."
    
    q = Queue()
    processes = [Process(target=specialized_agent, args=(q, base_query, perspective)) for perspective in perspectives]
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
        
    results = {}
    while not q.empty():
        try:
            result = q.get_nowait()
            results[result['sector']] = result['response']
        except Empty:
            break
    
    arbitrator_process = Process(target=arbitrator_agent, args=(q, results['cognitivism'], results['behaviorism']))
    arbitrator_process.start()
    arbitrator_process.join()
    
    results['arbitrator'] = q.get()['response']
    
    print("Results from specialized agents:")
    for key, value in results.items():
        print(f"{key}:\n{value}\n")