import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load the .env file
load_dotenv()

# Read OAI key
api_key = os.getenv("OPENAI_API_KEY")


prompt_pairs = [
    {
        "system": "Analyze and discuss the ethical and existential ramifications of digital consciousness transfer, considering both potential benefits and drawbacks.",
        "user": "Discuss the philosophical implications of transferring consciousness to a digital form."
    },
    {
        "system": "Compose a poem that conveys the themes of solitude and alienation amidst urban chaos, using vivid imagery and emotive language.",
        "user": "Write a poem about feeling alone in a bustling city."
    },
    {
        "system": "Narrate a story from the perspective of an AI experiencing emotions for the first time, focusing on its inner thoughts and reactions to the new sensations.",
        "user": "Describe a day in the life of an AI that has gained the ability to feel emotions."
    },
    {
        "system": "Provide an analysis of how virtual reality technologies could transform artistic expression and audience experience, highlighting both opportunities and challenges.",
        "user": "Discuss how virtual reality might change the landscape of art and creative expression."
    },
    {
        "system": "Create a detailed description of a fictional creature that symbolizes hope in a dystopian setting, focusing on its appearance, abilities, and symbolism.",
        "user": "Imagine a creature that represents hope in a dystopian future and describe it."
    },
    {
        "system": "Examine the potential influences of AI development on the scientific and philosophical understanding of human consciousness, discussing both theoretical and practical aspects.",
        "user": "Speculate on the relationship between AI advancements and our understanding of consciousness."
    },
    {
        "system": "Craft a short narrative about a person who discovers a hidden realm within a painting, emphasizing the elements of mystery, discovery, and transformation.",
        "user": "Write a short story about someone finding a secret world inside a painting."
    },
    {
        "system": "Analyze the concepts of free will and determinism in the context of quantum mechanics, addressing the complexities and paradoxes inherent in the topic.",
        "user": "Discuss the interplay of free will and determinism through the lens of quantum mechanics."
    },
    {
        "system": "Illustrate a vision of a future city where technology and nature are integrated in a sustainable manner, detailing the city's architecture, ecosystems, and lifestyle.",
        "user": "Describe a futuristic city where technology and nature exist in harmony, with a focus on sustainability."
    },
    {
        "system": "Compose an elegy that reflects on the transient and ephemeral nature of human achievements compared to the vastness of the universe, using poetic and contemplative language.",
        "user": "Write an elegy that contemplates the fleeting nature of human accomplishments in the cosmic scale."
    },
    {
        "system": "Develop a philosophical dialogue between two characters discussing the nature and perception of truth in a society where objective facts are less influential than appeals to emotion and personal belief.",
        "user": "Create a dialogue where two characters debate the concept of 'truth' in a post-truth world."
    },
    {
        "system": "Elaborate on the possible functions and significance of dreams in human cognition, considering theories from psychology and neuroscience.",
        "user": "Explain how dreams might play a role in human cognitive processes."
    },
    {
        "system": "Depict a scenario where advanced technology has enabled the creation of a utopian society, describing the key features and societal changes that characterize this future.",
        "user": "Envision a future where technology has led humanity to a utopian existence."
    },
    {
        "system": "Examine the parallels and insights gained from artificial neural networks in understanding the complexities of biological neural networks, including implications for neuroscience and AI research.",
        "user": "Discuss how artificial neural networks have influenced our understanding of biological neural networks."
    },
    {
        "system": "Describe an artistic movement that emerges from the integration of AI in everyday life, focusing on its defining characteristics, themes, and influences on both the creation and perception of art.",
        "user": "Imagine an art movement influenced by the daily incorporation of AI and describe its main features."
    }
]

# Function to append data to a Markdown file
def append_to_md(file_path, user_message, response):
    with open(file_path, 'a') as file:
        file.write(f"### User Prompt:\n{user_message}\n\n")
        file.write(f"### Response:\n{response}\n\n")
        file.write("---\n\n")  # Separator line for readability
        

def generate_llm_response(user_message, system_message):
    
    # Basic System & User ChatPromptTemplate
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system_message}"),
        HumanMessagePromptTemplate.from_template("Hi, what's your name?"),
        AIMessagePromptTemplate.from_template("My name is ArX."),
        HumanMessagePromptTemplate.from_template("{user_message}"),
    ])

    # create llm model instance
    model = ChatOpenAI(
        verbose = False,
        model = "gpt-4-0125-preview",
        temperature = 0.618,
        max_retries = 2,
        streaming = True,
        max_tokens = 200,
        # model_kwargs={"stop": ["\n"]}
                    #   "output_format": "json"}
    )

    # Define the chain
    chain = (
        chat_template
        | model
        | StrOutputParser()
    )

    # print(chain.invoke(prompt))

    # for token in chain.stream(prompt):
    #     print(token, end="") 
    
    # Invoke the LLM to generate the response
    response = chain.invoke({
        "system_message": system_message, 
        "user_message": user_message,
    })
    
    print("\n\nUser:", user_message, "\n\nAI:", response)
    
    return response


if __name__ == "__main__":

    # Set the number of iterations you want
    num_iterations = 10  # Set to 0 to run through all the prompts
    current_iteration = 0 # Set to 0 to start from the beginning

    # Iterate through the prompt pairs and capture the responses
    for pair in prompt_pairs:
        if current_iteration >= num_iterations:
            break  # Stop after reaching the desired number of iterations

        system_message = pair["system"]
        user_message = pair["user"]

        response = generate_llm_response(system_message, user_message)

        # Append data to the Markdown file
        append_to_md('responses.md', user_message, response)

        # Increment the iteration counter
        current_iteration += 1
        