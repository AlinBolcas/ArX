# textGen.py

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
short_memory_buffer_openai = ConversationTokenBufferMemory(llm=llm_openai, memory_key="short_memory_buffer_openai", return_messages=True, max_token_limit=8000)
short_memory_buffer_ollama = ConversationTokenBufferMemory(llm=llm_ollama, memory_key="short_memory_buffer_ollama", return_messages=True, max_token_limit=8000)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=684, chunk_overlap=50)

# FUNCTIONS
# ----------------------------------
def genText_simple_OpenAI(user, system, context, streaming=True):
    
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
    embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # PROMPT
    promptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="short_memory_buffer_openai"),
    HumanMessagePromptTemplate.from_template(user_message),
    ])

    # CHAINS
    chain = (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        | RunnablePassthrough.assign(short_memory_buffer_openai=RunnableLambda(short_memory_buffer_openai.load_memory_variables) | itemgetter("short_memory_buffer_openai"))
        | promptTemplate
        | llm_openai
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

def genText_simple_Ollama(user, system, context, streaming=True):
    
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
    embedding=OllamaEmbeddings(model="llama3"),
    )
    retriever = vectorstore.as_retriever()

    # PROMPT
    promptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="short_memory_buffer_ollama"),
    HumanMessagePromptTemplate.from_template(user_message),
    ])

    # CHAINS
    chain = (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        | RunnablePassthrough.assign(short_memory_buffer_ollama=RunnableLambda(short_memory_buffer_ollama.load_memory_variables) | itemgetter("short_memory_buffer_ollama"))
        | promptTemplate
        | llm_ollama
        | StrOutputParser()
    )
    
    if streaming:
        output = ""
        for token in chain.stream(user):
            yield token
            # print(token, end="", flush=True)
            output += token
        short_memory_buffer_ollama.save_context({"input": user}, {"output": output})
    else:
        output = chain.invoke(user)
        short_memory_buffer_ollama.save_context({"input": user}, {"output": output})
        return output

def textGen_promptGen(prompt):

    src_path = Path(__file__).parent.parent / 'src' 
    with open(str(src_path / 'image_prompts.json'), 'r') as file:
        image_prompts_data = json.load(file)

    # Read Markdown file (assuming it contains additional context or template)
    with open(str(src_path / 'image_sys_prompt.md'), 'r') as md_file:
        instructions = md_file.read()

    # Prepare messages for GPT-4 Completion
    messages = [
        {"role": "system", "content": f"You are a world-class creative visual artist, a master of photography, painting, lighting, composition, poetry and all arts. Use these instructions to generate image prompts: \n {instructions}"},
    ]

    # Adding prompts from JSON data
    for prompt in image_prompts_data:
        messages.append({"role": "user", "content": prompt["user"]})
        messages.append({"role": "assistant", "content": prompt["assistant"]})

    # Optionally, add content from markdown file
    messages.append({"role": "user", "content": f"Generate an image prompt in the same format as before in reply to this incentive: \n{prompt}"})

    print("Generating image prompt...")
    
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    max_tokens=150,
    temperature=1.2,
    )
    return response.choices[0].message.content

def textGen_vision(prompt):
    """
    Vision function for generating images from text prompts.
    Args:
        prompt (_type_): _description_
    """
    import base64
    from io import BytesIO

    from IPython.display import HTML, display
    from PIL import Image

    from langchain_community.llms import Ollama
    import os

    bakllava = Ollama(model="bakllava")

    def convert_to_base64(pil_image):
        """
        Convert PIL images to Base64 encoded strings

        :param pil_image: PIL image
        :return: Re-sized Base64 string
        """

        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")  # You can change the format if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


    def plt_img_base64(img_base64):
        """
        Disply base64 encoded string as image

        :param img_base64:  Base64 string
        """
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))


    file_path = "/Users/arvolve/Desktop/quote2.jpeg"
    pil_image = Image.open(file_path)
    image_b64 = convert_to_base64(pil_image)
    plt_img_base64(image_b64)

    llm_with_image_context = bakllava.bind(images=[image_b64])
    output = llm_with_image_context.invoke("expand on the idea seen in the image")


    print(output)

def image_necesity(input_text, outline):
    # self evaluates if an image would be applicable to be generated to support the given text input
    system_message = """
    You are a world class creative visual artist and writer with decades of experience in book illustration.
     in writing and editing content for books, magazines, and other publications.
    The book you're currently working on illustrating is has the outline:
    ---
    {context}
    """
    user_message = """
    Assert the applicability of generating an image for the given content to be introduced in the book.
    Based on the following provided text and the contextual outline, ansamble a directive for the image generation.
    This is the latest draft of a section from the book:
    ---
    {input}
    """
    
    @tool
    def gen_img_pipe(prompt: str) -> int:
        """
        Takes a text as input as a general directive for image generation,
        Refines the input prompt,
        Generates an image with dalle-3,
        And returns the refined prompt which describes the image.
        """
        def worker(prompt):
            img_prompt = imgen.gen_prompt(prompt)
            print("Image Prompt:\n", img_prompt, "\n")
            img_url = imgen.gen_img_dalle3(img_prompt)
            # If you have a GUI or other method to display or use the img_url, include it here
            # If using a callback to process the image URL:
            if callback:
                callback(img_url)

        # Define a callback function as needed
        def callback(img_url):
            imgen.display_image(img_url)
            imgen.save_image(img_url, "output/generated_image.jpg")
            print("\nGenerating Image succesfully...")
            return img_url
            # print("Image URL:", img_url)  # Placeholder for actual image processing/display

        # Start a new thread for each call to gen_img_pipe
        thread = Thread(target=worker, args=(prompt))
        thread.start()

    tools = [gen_img_pipe]

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

def textGen_summariser(prompt):
    """
    Summariser function for generating summaries of text.
    Args:
        prompt (_type_): _description_
    """
    pass


# OTHER WIP
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
        if self.model_provider == "OpenAI":
            self.llm = ChatOpenAI(
                verbose = True,
                model = "gpt-4-0125-preview",
                temperature = 0.618,
                max_retries = 3,
                streaming = True,
                max_tokens = 350,
                # model_kwargs={"stop": ["\n"]}
                            #   "output_format": "json"}
            )
        elif self.model_provider == "Ollama":
            # checking if the model is available, if not, pulling it
            try:
                ollama.chat(model)
            except ollama.ResponseError as e:
                print('Error:', e.content)
            if e.status_code == 404:
                ollama.pull(model)
            self.llm = ChatOllama(
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
        
        self.short_memory_buffer = ConversationTokenBufferMemory(llm=self.llm, memory_key="short_memory_buffer", return_messages=True, max_token_limit=8000)
        
        if self.image_URL:
            self.llm = self.llm.bind(images=[self.image_URL])
            # add vision model? 
            # response = llm_with_image_context.invoke(self.user_prompt)
    
    def textGen_universal(self):
        """Universal wrapper function for generating text with langchain and any provider API."""
        
        

# MAIN
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
        "how do I get become so rich that I never need to be worried about money?",
        "how do I best spend my remaining life at the age of 28?",
        "how do find a woman I can love and loves me if I'm a weird lonely misfit in this world with social anxiety?",
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
        
    for prompt in prompts3:
        print("\n", prompt, "\n")
        # prompt = input("Enter prompt: \n")
        # genText_Ollama(prompt, test_system)
        # print("\n", genText_OpenAI(prompt, bukowski_system, test_context))
        for token in genText_simple_Ollama(prompt, bukowski_system, test_context):
            print(token, end="", flush=True)