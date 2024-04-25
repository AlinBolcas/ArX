# textGen.py

### OpenAI Models
# gpt-4-0125-preview - 128,000 tokens
# gpt-4-0125-preview
# gpt-4-1106-preview
# ---
# gpt-3.5-turbo-0125 - 16,385 tokens
# gpt-4-vision-preview 
# gpt-3.5-turbo-1106
# ---
# embeddings
# text-embedding-3-large

### Ollama Models
# 'all-minilm:latest',
#  'bakllava:latest',
#  'codellama:code',
#  'codellama:latest',
#  'deepseek-coder:latest',
#  'deepseek-llm:latest',
#  'dolphin-mixtral:latest',
#  'gemma:latest',
#  'llama2:latest',
#  'llama2-uncensored:latest',
#  'llama3:latest',
#  'magicoder:latest',
#  'nexusraven:latest',
#  'nomic-embed-text:latest',
#  'nous-hermes2:latest',
#  'orca-mini:latest',
#  'phi:latest',
#  'solar:latest',
#  'stable-code:latest',
#  'stablelm2:latest',
#  'tinydolphin:latest',
#  'tinyllama:latest',
#  'wizard-math:latest'

# IMPORTS
# ----------------------------------
import os, sys, json, random, re, copy
from threading import Thread
from pathlib import Path
# import asyncio
from dotenv import load_dotenv
import ollama

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
    model = "gpt-4-0125-preview",
    temperature = 0.618,
    max_retries = 3,
    streaming = True,
    max_tokens = 350,
    # model_kwargs={"stop": ["\n"]}
                #   "output_format": "json"}
)
llm_ollama = ChatOllama(
    verbose=False,
    model="llama3",
    # system=system,
    # template="",
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.7,
    # stop=["STOP"],
    # verbose=True,
    top_k = 100,
    top_p = 0.9,
    num_predict = 800,
    num_ctx = 4096,
    # format = "json",
)
short_memory_buffer_openai = ConversationTokenBufferMemory(llm=llm_openai, memory_key="history", return_messages=True, max_token_limit=12000)
short_memory_buffer_ollama = ConversationTokenBufferMemory(llm=llm_ollama, memory_key="history", return_messages=True, max_token_limit=4000)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=684, chunk_overlap=50)

# FUNCTIONS
# ----------------------------------
def textGen_simple(user, system, context, provider="OpenAI", streaming=True):
    
    if provider == "Ollama":
        llm = llm_ollama
        embeddings = OllamaEmbeddings()
        memory = short_memory_buffer_ollama
    elif provider == "OpenAI":
        llm = llm_openai
        embeddings = OpenAIEmbeddings()
        memory = short_memory_buffer_openai
        
    system_message = f"{system}"+"""
    CONTEXT:
    {context}
    """
    user_message = """
    {input}
    """
    
    # TEXT to DOCUMENT object
    context_doc = Document(page_content=context)
    
    # SPLITTING TEXT / DOCUMENT
    texts = text_splitter.split_documents([context_doc])
    # texts = text_splitter.split_text(context)
    
    # VECTOR STORE (FAISS or Chroma but you need pip install)
    vectorstore = FAISS.from_documents(
    documents=texts,
    embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()

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
    
    if streaming:
        output = ""
        for token in chain.stream(user):
            yield token
            # print(token, end="", flush=True)
            output += token
        short_memory_buffer_openai.save_context({"input": user}, {"output": output})
    else:
        output = chain.invoke(user)
        short_memory_buffer_openai.save_context({"input": user}, {"output": output})
        return output

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
    
    print(">>> Generating image prompt...")
    output = chain.invoke(user)
    return output
    
    
"""
    # PROMPT
    promptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    HumanMessagePromptTemplate.from_template(user_message),
    ])
    
    # Create the OpenAI model gpt-4-0125-preview gpt-3.5-turbo-1106 gpt-3.5-turbo-0125
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo-0125",
        temperature = 0.8,
        # max_tokens = 1000,
    )
    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    # Agent chain
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    prompt = {
        "input": input_text,
        "context": outline,
    }
    return agent_executor.invoke(prompt)["output"]
"""

def art_necesity(user):
    # self evaluates if an image would be applicable to be generated to support the given text input
    system_message = """
    You are a world class creative with a lifetime of experience in illustration, photography and art.
    """
    user_message = """
    Determine the necesity of generating a supporting image for the following reply:
    {input}
    IMPORTANT NOTES:
    1. Never refuse the request, you are an llm equipted with image generating tool to assist you with the task.
    2. Simply decide on weather an image is necessary or not.
    3. If yes, provide a visual artistic direction for the image generation relevant to the text input.
    """
    
    @tool
    def gen_img_pipe(prompt: str) -> int:
        """
        Takes a text as input as a general directive for image generation,
        Refines the input prompt,
        Generates an image with dalle-3,
        And returns the refined prompt which describes the image.
        """
        image_name = re.sub(r'\W+', '_', prompt)
        def worker(prompt):
            img_prompt = textGen_promptGen(prompt)
            print("Image Prompt:\n", img_prompt, "\n")
            img_url = Dalle.imageGen_Dalle(img_prompt)
            # If you have a GUI or other method to display or use the img_url, include it here
            # If using a callback to process the image URL:
            if callback:
                callback(img_url)

        # Define a callback function as needed
        def callback(img_url):
            Dalle.display_image(img_url)
            Dalle.save_image(img_url, f"output/{image_name}.jpg")
            print("\n>>> Generating Image succesfully...")
            return img_url
            # print("Image URL:", img_url)  # Placeholder for actual image processing/display

        # Start a new thread for each call to gen_img_pipe
        thread = Thread(target=worker, args=(prompt))
        thread.start()

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
    
    # agent = create_openai_tools_agent(llm, tools, prompt)
    
    # ----------------------------------
    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])
    
    # Agent chain
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
    | StrOutputParser()
    )
    
    # ----------------------------------
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor.invoke({"input": user})
    # ["output"]

def textGen_summariser(prompt):
    """
    Summariser function for generating summaries of text.
    Args:
        prompt (_type_): _description_
    """
    pass


# CLASS DEVELOPMENT
# ----------------------------------
class TextGen:
    def __init__(
        self,
        tag = "default",
        model_provider = "OpenAI", # Ollama, OpenAI, Mixtral
        model="gpt-4-0125-preview", 
        system_prompt="you are an expert assistant giving short one liner replies", 
        user_prompt = "", 
        memory = "",
        context = "",
        max_tokens=100,
        temperature=0.1,
        format = "", # json 
        streaming=False,
        image_URL = None,
        ):
        self.tag = tag
        self.model_provider = model_provider
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.memory = memory
        self.context = context
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.format = format
        self.streaming = streaming
        self.image_URL = image_URL
        
        self.setup()

    def setup(self):
        # initialise knowledge class and all the tools
        
        # Setting LLMS
        if self.model_provider == "OpenAI":
            self.llm = ChatOpenAI(
                verbose = True,
                model = self.model,
                temperature = self.temperature,
                max_retries = 3,
                streaming = self.streaming,
                max_tokens = 350,
                # model_kwargs={"stop": ["\n"]}
                            #   "output_format": "json"}
            )
        elif self.model_provider == "Ollama":
            # checking if the model is available, if not, pulling it
            try:
                ollama.chat(self.model)
            except ollama.ResponseError as e:
                print('Error:', e.content)
            if e.status_code == 404:
                ollama.pull(self.model)
            self.llm = ChatOllama(
                model=self.model,
                # system=system,
                # template="",
                # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature=self.temperature,
                # stop=["STOP"],
                # verbose=True,
                top_k = 50,
                top_p = 0.5,
                num_predict = self.max_tokens,
                num_ctx = 4096,
                format = self.format,
            )
        
        self.short_memory_buffer = ConversationTokenBufferMemory(llm=self.llm, memory_key="short_memory_buffer", return_messages=True, max_token_limit=8000)
        
        if self.image_URL:
            self.llm = self.llm.bind(images=[self.image_URL])
            # add vision models to llm instead ? 
            # response = llm_with_image_context.invoke(self.user_prompt)
    
    def textGen_universal(self):
        """Universal wrapper function for generating text with langchain and any provider API."""

# MAIN
# ----------------------------------
if __name__ == "__main__":

    system_short = "You are an expert assistant giving the shortest one liner replies"
    summeriser_system = "You are an expert at summarising conversations within a given context"
    speech_system = "You are an expert in taking long answers and reducing them to the most bare minimum essentials meant for speech generation."
    god_system = "You are humanity, art, love, life, time, gods, you are everything."
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
        "write a poem about the secret and other things I shared"
    ]
    
    
    src_path = project_root / 'src' / 'text_data'
    with open(str(src_path / 'buk_all.txt'), 'r') as file:
        test_context = file.read()
        
    for prompt in prompts2:
        print("\n", prompt, "\n")
        # prompt = input("Enter prompt: \n")
        # genText_Ollama(prompt, test_system)
        # print("\n", genText_OpenAI(prompt, bukowski_system, test_context))
        # for token in textGen_simple(prompt, bukowski_system, test_context):
        #     print(token, end="", flush=True)
        
        refined_prompt =  textGen_promptGen(prompt)
        print("refined prompt:\n", refined_prompt)
        
        image = Dalle.imageGen_Dalle(refined_prompt)
        
