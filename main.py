import getpass
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your API key here:")


messages = [
    (
        "system",
        '''
        You are a helpful AI research agent that helps people find out information on the topic that 
        they've asked about. You have the access to use multiple tools so as to provide the most accurate and
        precise response to the user, regarding the nature of the query that the user has asked. 
        
        '''
    ),

    ("human",
     '''
        What is Maxwell's equations of ElectroMagnetism. State their historical importance. 
     '''),
]

# Setting up an instance of User's Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = ChatGroq(
    model = "llama-3.1-8b-instant",
    messages=messages

)


