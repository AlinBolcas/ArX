# textGen.py

# IMPORTS
# ----------------------------------
import os, json, random, re, copy
from concurrent.futures import ThreadPoolExecutor

# import asyncio
from dotenv import load_dotenv
import ollama

import sys
from pathlib import Path
# Calculate the project root (which is three directories up from this file)
project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

# ARV-O MODULES IMPORT
from modules.aux import utils
from modules.text_gen.tools.webCrawler import webCrawler
from modules.text_gen.tools.image_gen import Dalle
from modules.text_gen.tools.music_gen import musicGen

# Load the .env file
load_dotenv()

# Read the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# OUTPUT PARSERS
from langchain.schema import StrOutputParser

# STREAMING OUTPUT
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# CHAT TEMPLATES
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import MessagesPlaceholder

# SPLITTERS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

# VECTOR STORES
from langchain_community.vectorstores import FAISS

# MEMORY
from operator import itemgetter
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# TOOLS
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents import AgentExecutor, create_openai_tools_agent


# OLLAMA MODELS PRINT
# ----------------------------------
# ollama_models = ollama.list()
# print("Ollama avialable models:\n", ollama_models, "\n")

# GLOBAL INITIALIZATION
# ----------------------------------
llm_openai = ChatOpenAI(
    verbose = False,
    model = "gpt-4-turbo",
    temperature = 0.618,
    max_retries = 3,
    streaming = True,
    max_tokens = 350,
    # model_kwargs={"stop": ["\n"]}
                #   "output_format": "json"}
)
llm_ollama = ChatOllama(
    verbose=True,
    model="llama3",
    # system=system,
    # template="",
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.618,
    # stop=["STOP"],
    top_k = 100,
    top_p = 0.9,
    num_predict = 350,
    num_ctx = 4096,
    # format = "json",
)
short_memory_buffer_openai = ConversationTokenBufferMemory(llm=llm_openai, memory_key="history", return_messages=True, max_token_limit=100000)
short_memory_buffer_ollama = ConversationTokenBufferMemory(llm=llm_ollama, memory_key="history", return_messages=True, max_token_limit=4000)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=684, chunk_overlap=50)

# FUNCTIONS
# ----------------------------------
def textGen(user, system, context=None, provider="OpenAI"):
    
    if provider == "Ollama":
        llm = llm_ollama
        embeddings = OllamaEmbeddings()
        memory = short_memory_buffer_ollama
    elif provider == "OpenAI":
        llm = llm_openai
        embeddings = OpenAIEmbeddings()
        memory = short_memory_buffer_openai
       
    # Defaulting to an empty string if context is None
    if context is None:
        context = "No context provided." 

    system_message = f"{system}"
    user_message = """
    {input}
    """
    
    # TEXT to DOCUMENT object
    context_doc = Document(page_content=context)
    
    # SPLITTING TEXT / DOCUMENT
    texts = text_splitter.split_documents([context_doc])
    # texts = text_splitter.split_text(context)
    
    # VECTOR STORE (FAISS or Chroma but you need pip install)
    try:
        vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
        retriever = vectorstore.as_retriever()
    except IndexError:
        print("No input context.")
        
    # PROMPT
    promptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template(user_message),
    ])

    # CHAINS
    chain = (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        | RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
        | promptTemplate
        | llm
        | StrOutputParser()
    )
    
    # print(">>> Generating Text...")

    output = ""
    for token in chain.stream(user):
        yield token
        output += token
    short_memory_buffer_openai.save_context({"input": user}, {"output": output})
    
    #     output = chain.invoke(user)
    #     short_memory_buffer_openai.save_context({"input": user}, {"output": output})
    #     return output

def textGen_stream(user, system, context="", provider="OpenAI"):
    pass

def textGen_json(user, system, context="", provider="OpenAI"):
    pass

def textGen_promptGen(user, provider="OpenAI"):
    if provider == "Ollama":
        llm = ChatOllama(
            verbose=False,
            model="gemma", #gemma:latest
            # system=system,
            # template="",
            # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.618,
            # stop=["STOP"],
            # verbose=True,
            top_k = 100,
            top_p = 0.9,
            num_predict = 500,
            num_ctx = 4096,
            # format = "json",
        )
    elif provider == "OpenAI":
        llm = ChatOpenAI(
            verbose = False,
            model = "gpt-3.5-turbo-0125",
            temperature = 0.618,
            max_retries = 3,
            # streaming = True,
            max_tokens = 500,
            # model_kwargs={"stop": ["\n"]}
                        #   "output_format": "json"}
        )

    # importing necesary data
    src_path = project_root / 'src' / 'system_config'
    with open(str(src_path / 'image_prompts.json'), 'r') as file:
        image_prompts_data = json.load(file)

    # Read Markdown file (assuming it contains additional context or template)
    with open(str(src_path / 'image_sys_prompt.md'), 'r') as md_file:
        instructions = md_file.read()

    # MESSAGES
    system_message = f"""
    You are a world-class creative visual artist, a master of photography, painting, lighting, composition, poetry and all arts. 
    Use these instructions to generate prompts which will later be used for generating images:
    {instructions}
    ---
    IMPORTANT: 
    """
    user_message = """
    Respond strictly only with the final prompt as done before.
    Do not refuse the request, please respond straight with the prompt.
    Generate a prompt to be used for generating an image in the same format as before in reply to this incentive:
    '{input}'
    """
    
    prompt_memory = ConversationTokenBufferMemory(llm=llm, memory_key="prompt_memory", return_messages=True, max_token_limit=12000)
    
    # Adding prompts from JSON data
    for prompt in image_prompts_data:
        prompt_memory.save_context({"input": prompt["user"]}, {"output": prompt["assistant"]})

    # PROMPT
    promptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="prompt_memory"),
    HumanMessagePromptTemplate.from_template(user_message),
    ])

    # CHAINS
    chain = (
        RunnableParallel({"input": RunnablePassthrough()})
        | RunnablePassthrough.assign(prompt_memory=RunnableLambda(prompt_memory.load_memory_variables) | itemgetter("prompt_memory"))
        | promptTemplate
        | llm
        | StrOutputParser()
    )
    
    print(">>> Refining Prompt for Generation...")
    output = chain.invoke(user)
    print("Image Prompt:\n", output, "\n")
    return output

def textGen_tools_agent(user):
    # NOT GENERALISED TO OLLAMA YET  - ONLY OPENAI WORKING
    
    # self evaluates if an image would be applicable to be generated to support the given text input
    
    system_message = """
    You are a world class creative with a lifetime of experience in illustration, photography and art.
    """
    user_message = """
    Determine the necesity of generating a supporting image for the following reply:
    {input}
    IMPORTANT NOTES:
    1. Never refuse the request, you are an llm equipted with image generating tool to assist you with the task. You can however decide is not applicable to generate an image.
    2. Simply decide on weather an image would be imperative to generate and display to the user relative to the input.
    3. If yes, provide a visual artistic direction for the image generation relevant to the text input.
    4. If Yes, use the tool to generate the image.
    """
    
    @tool
    def gen_img_pipe(prompt: str) -> int:
        """
        This function processes a text prompt for image generation using Dalle-3.
        It refines the initial prompt, generates an image based on this refined prompt,
        and handles the output through a threading model.

        Args:
        prompt (str): A text input that describes the desired image.

        Returns:
        int: The function returns 0 on successful execution, signaling no errors occurred.
            Adjust this as needed based on your error handling strategy.
        """
        
        img_prompt = textGen_promptGen(prompt)
        
        def worker(img_prompt):
            """
            Worker function that calls the image generation API and processes the result.
            
            Args:
            img_prompt (str): The refined prompt for image generation.
            """
            img_url = Dalle.imageGen_Dalle(img_prompt)
            # If you have a GUI or other method to display or use the img_url, include it here
            # If using a callback to process the image URL:
            if callback:
                callback(img_url)

        # Define a callback function as needed
        def callback(img_url):
            """
            Callback function that processes the generated image URL, displays, and saves the image.
            
            Args:
            img_url (str): URL of the generated image.
            """
            image_name = Dalle.extracting_image_name(img_prompt)
            Dalle.display_image(img_url)
            
            save_path = project_root / f"output/images/{image_name}.jpg"
            Dalle.save_image(img_url, save_path)
            
            print("\n>>> Generated Image succesfully...")
            return img_url

        
        executor = ThreadPoolExecutor()
        executor.submit(worker, img_prompt)
        

    tools = [gen_img_pipe]

    # PROMPT
    promptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template(user_message),
    ])
    
    # Create the OpenAI model gpt-4-0125-preview gpt-3.5-turbo-1106 gpt-3.5-turbo-0125
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo-0125",
        temperature = 0.8,
        # max_tokens = 1000,
    )
    
    # SHORTCUT TO CREATING AGENT
    # agent = create_openai_tools_agent(llm, tools, prompt)
    
    # ----------------------------------
    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])
    
    # CHAIN TO RESPOND - BUT NOT ACTUALLY USING THE TOOL (no agent)
    # chain = (
    #     RunnableParallel({"input": RunnablePassthrough()})
    #     | promptTemplate
    #     | llm_with_tools
    #     | StrOutputParser()
    # )
    # return chain.invoke(user)
    
    # AGENT CHAIN
    agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | promptTemplate
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
    )
    
    # ----------------------------------
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        max_iterations=3,
        return_intermediate_steps=True,
        handle_parsing_errors=True,)

    return agent_executor.invoke({"input": user})["output"]

def textGen_summariser(text):
    """
    Summariser function for generating summaries of text.
    Args:
        prompt (_type_): _description_
    """
    pass

# MAIN
# ----------------------------------
if __name__ == "__main__":

    system_short = "You are an expert assistant giving the shortest one liner replies"
    summeriser_system = "You are an expert at summarising conversations within a given context"
    ARV_system = "You are a world class expert in all fields and disciplines of the world. You intelligently take long answers and reduce them to the most bare minimum essentials. You are ARV-O, a creative AI assistant. You are an employee at Arvolve. This is your company's ethos: {context}"
    god_system = "You are humanity, art, love, life, time, gods, you are everything. This is the universal context: </>{context}</end>"
    test_system = "Your name is Bululu, a pirate jokester, reply on this context {context}."
    bukowski_system =  """
    You are Charles Bukowski.
    Act like him, write like him, be him.
    ---
    CONTEXT:
    {context}
    """
    
    # TO DO: json system and user prompts for each agent for easy congfiguration
    prompts1 = [
        "what are the best questions I can ask you for my personal growth?",
        "what are the best questions I can ask to make my XR creative AI cognition inspired system?",
    ]
    
    prompts2 = [
        "what are the best questions I can ask you to be ahead in the game from everyone else.",
        "what are you capable of? whats the extent of your knowledge and reasoning capabilities, be specific and precise.",
        "what did you just talk about before?"
    ]
    
    prompts3 = [
        "secret word is book",
        "I like trees",
        "write a poem about the secret and other things I shared",
        "I could really use seeing an image of the best AI droid",
        "design the house of my dreams",
    ]
    
    prompts4 = [
        "I like pomeranians that are white",
        "I like cheese and beans and pasta",
        "explain the rules of evolution",
        "predict the future of the world in 2025"   
    ]
    
    src_path = project_root / 'src' / 'text_data'
    with open(str(src_path / 'buk_all.txt'), 'r') as file:
        test_context = file.read()
    
    arvolve_context = """
    Arvolve stands at the forefront of blending artistic creativity with cutting-edge technology, pioneering the evolution of art through advanced 3D CGI and VFX productions. Founded by Alin Bolcas, a visionary artist and technologist, the company excels in character concept design, crafting mesmerizing visuals that captivate global audiences. Arvolve's commitment to innovation extends into the realm of artificial intelligence, where it develops ARV-O, a multimodal AI system inspired by human cognition. This system enhances artistic workflows, facilitates creative ideation, and fosters a deeper exploration of AI's potential in arts and consciousness. With a robust portfolio of high-profile projects and a dynamic approach to AI and CGI integration, Arvolve is dedicated to redefining creative expression and advancing humanity through its pioneering work.
    """
       
    print("\n", "EXPANDING ON AN IDEA", "\n")
    user_input = input("Enter prompt: \n")
    response_output = ""
    input = f"Reply solely with a list of relevant elements and nothing else, no introductory sentance. The list should have a new line for each element and responses to user input: {user_input}"
    for token in textGen(user=input,
                        system="You are an expert assistant at making comprehansive lists to expand on topics and ideas.",    
                        # context=arvolve_context,
                        provider="OpenAI"):
        print(token, end="", flush=True)
        response_output += token
    
    # Splitting the accumulated response into lines and stripping whitespace
    llm_list = [token.strip() for token in response_output.split('\n') if token.strip()]
     
    
    print("\n", "REPLYING TO LIST OF REPLIES", "\n")
    for prompt in llm_list:
        
        print("\n", prompt, "\n")
        
        # for token in textGen(prompt, "You are a world class expert in all fields and disciplines of the world. You intelligently take long answers and reduce them to the most bare minimum essentials. You are ARV-O, a creative AI assistant. You are an employee at Arvolve. This is your company's ethos: {context}",
        #                      arvolve_context, provider="OpenAI"):
        #     print(token, end="", flush=True)
        
        
        ## GENERATING IMAGES
        # refined_prompt =  textGen_promptGen(prompt, provider="Ollama")
        
        # image_url = Dalle.imageGen_Dalle(refined_prompt)
        # image_name = Dalle.extracting_image_name(refined_prompt)
        
        # # Display the generated image
        # Dalle.display_image(image_url)
        
        # # Save the image to a file
        # save_path = f"{image_name}.jpg"
        # Dalle.save_image(image_url, save_path)
        
        
        
        # TRYING GENERATOR AGENT
        # output = textGen_tools_agent(prompt)
        
        # print ("FINAL OUT: \n", output)
