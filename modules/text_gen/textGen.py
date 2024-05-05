# textGen.py

# IMPORTS
# ----------------------------------
import os, json, random, re, copy
from concurrent.futures import ThreadPoolExecutor, as_completed

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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain_community.embeddings import OllamaEmbeddings

# OUTPUT PARSERS
from langchain.schema import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

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
import json

# CLASS DEVELOPMENT
# ----------------------------------
        
class TextGen:
    def __init__(
        self,
        model_provider = "OpenAI", # Ollama, OpenAI, Mixtral
        model="gpt-3.5-turbo", 
        system_prompt="you are an expert assistant giving short one liner replies", 
        max_tokens=100,
        temperature=0.1,
        vision_bool = False
        ):
        self.model_provider = model_provider
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.vision_bool = vision_bool
        
        self.init_setup()

    def init_setup(self):
        """Setup the language model based on the provider."""
        self.load_src_data()
        
        # checking for image input
        if self.vision_bool:
            self.model = "gpt-4-turbo"
            # add vision models to llm instead ?
            # response = llm_with_image_context.invoke(self.user_prompt)
        
        self.load_llm()
        # self.tools = tools.Tools()
        # self.knowledge = knowledge.Knowledge()
        
        # Short time memory buffer
        self.short_memory_buffer = ConversationTokenBufferMemory(llm=self.llm, memory_key="short_memory_buffer", return_messages=True, max_token_limit=8000)
        # add entity and graph memory ? >> to be stored somehow in the knowledge meta memory class?
        self.promptGen_memory = ConversationTokenBufferMemory(llm=self.llm, memory_key="promptGen_memory", return_messages=True, max_token_limit=12000)
        # Adding prompts from JSON data
        for prompt in self.image_prompts:
            self.promptGen_memory.save_context({"input": prompt["user"]}, {"output": prompt["assistant"]})

        # Text splitter
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=684, chunk_overlap=50)
        # add reccursive and semantic text splitters
        
    def load_src_data(self):
        """Load the source data for the context."""
        src_path = project_root / 'src' / 'text_data'
        with open(str(src_path / 'buk_all.txt'), 'r') as file:
            self.buk_context = file.read()
 
        src_path = project_root / 'src' / 'system_config'
        with open(str(src_path / 'promptTemplates.json'), 'r') as file:
            self.promptTemplates = json.load(file)       
        
        src_path = project_root / 'src' / 'system_config'
        with open(str(src_path / 'image_prompts.json'), 'r') as file:
            self.image_prompts = json.load(file)
        with open(str(src_path / 'image_sys_prompt.md'), 'r') as md_file:
            self.image_instructions = md_file.read()

    def update_parameters(self, model_provider, model, system_prompt, max_tokens, temperature, vision_bool=False):
        """Update the parameters of the models"""
        self.model_provider = model_provider
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.vision_bool = vision_bool
        
        self.load_llm()
    
    def load_llm(self):
        # Setting LLMS
        if self.model_provider == "OpenAI":
            self.embeddings = OpenAIEmbeddings()
            
            self.llm_prompter = ChatOpenAI(
                verbose = True,
                model = self.model,
                temperature = self.temperature,
                max_retries=3,
                max_tokens = 300,
            )
            self.llm = ChatOpenAI(
                verbose = True,
                model = self.model,
                temperature = self.temperature,
                max_retries = 3,
                # streaming = self.streaming,
                max_tokens = self.max_tokens,
            )
        elif self.model_provider == "Ollama":
            # checking if the model is available, if not, pulling it
            try:
                ollama.chat(self.model)
            except ollama.ResponseError as e:
                print('Error:', e.content)
                if e.status_code == 404:
                    ollama.pull(self.model)
            
            self.embeddings = OllamaEmbeddings(model=self.model)
            
            self.llm_prompter = ChatOllama(
                model = self.model,
                temperature = self.temperature,
                num_predict = self.max_tokens,
            )
            self.llm = ChatOllama(
                model=self.model,
                # template="",
                # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                temperature=self.temperature,
                # stop=["STOP"],
                # verbose=True,
                # top_k = 50,
                # top_p = 0.5,
                num_predict = self.max_tokens,
                # num_ctx = 4096,
            )
        
    def textGen(self, user_input, context, json_bool=False, image=None):
        """Universal wrapper function for generating text with langchain and any provider API."""

        # Defaulting to an empty string if context is None
        if context is None:
            context = "No context provided." 

        if json_bool:
            system_message = f"{self.system_prompt}" + """
            ALWAYS REPLY IN JSON FORMAT!
            ---
            CONTEXT:
            {context}
            """
            output_parser = JsonOutputParser()
        else:
            system_message = f"{self.system_prompt}" + """
            ---
            CONTEXT:
            {context}
            """
            output_parser = StrOutputParser()
        
        # self.llm = self.llm.bind(images=[self.image_URL])
        
        user_message = """
        {input}
        """
        
        # TEXT to DOCUMENT object
        context_doc = Document(page_content=context)
        
        # SPLITTING TEXT / DOCUMENT
        texts = self.text_splitter.split_documents([context_doc])
        # texts = text_splitter.split_text(context)
        
        # VECTOR STORE (FAISS or Chroma but you need pip install)
        vectorstore = FAISS.from_documents(documents=texts, embedding=self.embeddings)
        retriever = vectorstore.as_retriever()
            
        # PROMPT
        promptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="short_memory_buffer"),
        HumanMessagePromptTemplate.from_template(user_message),
        ])

        # CHAINS
        chain = (
            RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
            | RunnablePassthrough.assign(short_memory_buffer=RunnableLambda(self.short_memory_buffer.load_memory_variables) | itemgetter("short_memory_buffer"))
            | promptTemplate
            | self.llm
            | output_parser
        )

        output = chain.invoke(user_input)
        
        self.short_memory_buffer.save_context({"input": user_input}, {"output": str(output)})
        return output
    
    def textGen_stream(self, user_input, context, image=None):
        """Universal STREAMING wrapper function for generating text with langchain and any provider API."""

        # Defaulting to an empty string if context is None
        if context is None:
            context = "No context provided." 

        system_message = f"{self.system_prompt}" + """
        CONTEXT:
        {context}
        """
        
        user_message = """
        {input}
        """
        
        # TEXT to DOCUMENT object
        context_doc = Document(page_content=context)
        
        # SPLITTING TEXT / DOCUMENT
        texts = self.text_splitter.split_documents([context_doc])
        # texts = text_splitter.split_text(context)
        
        # VECTOR STORE (FAISS or Chroma but you need pip install)
        vectorstore = FAISS.from_documents(documents=texts, embedding=self.embeddings)
        retriever = vectorstore.as_retriever()
            
        # PROMPT
        promptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="short_memory_buffer"),
        HumanMessagePromptTemplate.from_template(user_message),
        ])


        # CHAINS
        chain = (
            RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
            | RunnablePassthrough.assign(short_memory_buffer=RunnableLambda(self.short_memory_buffer.load_memory_variables) | itemgetter("short_memory_buffer"))
            | promptTemplate
            | self.llm
            | StrOutputParser()
        )

        output = ""
        for token in chain.stream(user_input):
            yield token
            output += token
        self.short_memory_buffer.save_context({"input": user_input}, {"output": output})

    def textGen_promptGen(self, user_input, image=None):

        # MESSAGES
        system_message = f"""
        You are a world-class creative visual artist, a master of photography, painting, lighting, composition, poetry and all arts. 
        Use these instructions to generate prompts which will later be used for generating images:
        {self.image_instructions}
        ---
        IMPORTANT: 
        """
        user_message = """
        Respond strictly only with the final prompt as done before.
        Do not refuse the request, please respond straight with the prompt.
        Generate a prompt to be used for generating an image in the same format as before in reply to this incentive:
        '{input}'
        """

        # PROMPT
        promptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="promptGen_memory"),
        # MessagesPlaceholder(variable_name="short_memory_buffer"),
        HumanMessagePromptTemplate.from_template(user_message),
        ])

        llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        temperature = 0.8,
        # max_tokens = 1000,
        )

        # CHAINS
        chain = (
            RunnableParallel({"input": RunnablePassthrough()})
            | RunnablePassthrough.assign(promptGen_memory=RunnableLambda(self.promptGen_memory.load_memory_variables) | itemgetter("promptGen_memory"))
            | promptTemplate
            | self.llm_prompter
            | StrOutputParser()
        )
        
        print(">>> Refining Prompt for Generation...")
        output = chain.invoke(user_input)
        print("Image Prompt:\n", output, "\n")
        return output

    def agent_imgGen(self, user_input):
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
        2. Simply decide on weath sr an image would be imperative to generate and display to the user relative to the input.
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
            
            img_prompt = self.textGen_promptGen(prompt)
            
            def worker(img_prompt):
                """
                Worker function that calls the imag
                e generation API and processes the result.
                
                Args:
                img_prompt (str): The refined prompt for image generation.
                """
                save_path = Dalle.imageGen_fullPipeline(img_prompt)
                # If you have a GUI or other method to display or use the img_url, include it here
                # If using a callback to process the image URL:
                if callback:
                    callback(save_path)

            # Define a callback function as n seeded
            def callback(save_path):
                """
                Callback function that processes the generated image URL, displays, and saves the image.
                
                Args:
                save_path (str): path to the saved image.
                """
                print(">>> Image generated! SAVED PATH:", save_path)
                return save_path

            executor = ThreadPoolExecutor()
            executor.submit(worker, img_prompt)
            
        tools = [gen_img_pipe]

        # PROMPT
        promptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(user_message),
        ])
        
        llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
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

        return agent_executor.invoke({"input": user_input})["output"]


    # def textGen_summariser(user, context):

# MAIN
# ----------------------------------
if __name__ == "__main__":

    system_short = "You are an expert assistant giving the shortest one liner replies"
    system_summariser = "You are an expert at summarising conversations within a given context"
    system_arvolve = "You are a world class expert in all fields and disciplines of the world. You intelligently take long answers and reduce them to the most bare minimum essentials. You are ARV-O, a creative AI assistant. You are an employee at Arvolve. This is your company's ethos: {context}"
    system_god = "You are humanity, art, love, life, time, gods, you are everything. This is the universal context: </>{context}</end>"
    system_test = "Your name is Bululu, a pirate jokester, reply on this context {context}."
    system_bukowski =  """
    You are Charles Bukowski. s
    Act like him, write like him, be him.
    ---
    CONTEXT:
    {context}
    """
    
    # TO DO: json system and user prompts for each agent for easy congfiguration
    prompts1 = [
        "what are the best questions I can ask you for my personal growth?",
        "what are the best questions I can ask to make my XR creative AI cognition inspired system?",
        "what did we jsut talk about?",
        "how do I best progress through the journey of programming?",
        "will 3d modelling be made redundant once AI can generate 3D models?",
        "how do we humans derive meaning in a fleeting world?",
        "how can one best enjoy and make the most out of life when the very nature of one's existing is conflicting with that mode of being?",
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
    You are talking with Alin Bolcas, the founder of Arvolve.
    He is developing you right now as you are in testing.
    Arvolve stands at the forefront of blending artistic creativity with cutting-edge technology, pioneering the evolution of art through advanced 3D CGI and VFX productions. 
    Alin Bolcas is a visionary artist and technologist, the company excels in character concept design, crafting mesmerizing visuals that captivate global audiences. 
    Arvolve's commitment to innovation extends into the realm of artificial intelligence, where it develops ARV-O, a multimodal AI system inspired by human cognition. 
    This system enhances artistic workflows, facilitates creative ideation, and fosters a deeper exploration of AI's potential in arts and consciousness. 
    With a robust portfolio of high-profile projects and a dynamic approach to AI and CGI integration, Arvolve is dedicated to redefining creative expression and advancing humanity through its pioneering work.
    """ 
    
    print("\n", "ANSWERS: ", "\n")
    for prompt in prompts2:
        
        print("\n\n USER PROMPT: ",prompt, "\n")
        
        # for token in textGen(prompt, "You are a world class expert in all fields and disciplines of the world. You intelligently take long answers and reduce them to the most bare minimum essentials. You are ARV-O, a creative AI assistant. You are an employee at Arvolve. This is your company's ethos: {context}",
        #                      arvolve_context, provider="OpenAI"):
        #     print(token, end="", flush=True)
        
        textGen1 = TextGen(
            model_provider="Ollama", 
            model="llama3", 
            system_prompt=system_arvolve, 
            max_tokens=300,
            temperature=0.7)
        
        # # STREAMING
        # print(" AI STREAM:")
        # for token in textGen1.textGen_universal_stream(prompt, arvolve_context):
        #     print(token, end="", flush=True)

        # # INVOKE
        # # JSON OUTPUT TEST
        # output = textGen1.textGen_universal(prompt, arvolve_context, json_bool=True)
        # # Convert the output to JSON format
        # output_json = json.dumps(output)
        # # Print the JSON output
        # print("\nAI INVOKE:\n", output_json)
        
        
        # PROMPTGEN IMAGES
        # refined_prompt =  textGen1.textGen_promptGen(prompt)
        
        # TRYING GENERATOR AGENT
        output = textGen1.agent_imgGen(prompt)
        # print(output)
        
        # image_url = Dalle.imageGen_Dalle(refined_prompt)
        # image_name = Dalle.extracting_image_name(refined_prompt)
        
        # # Display the generated image
        # Dalle.display_image(image_url)
        
        # # Save the image to a file
        # save_path = f"{image_name}.jpg"
        # Dalle.save_image(image_url, save_path)
        
        # print ("FINAL OUT: \n", output)
