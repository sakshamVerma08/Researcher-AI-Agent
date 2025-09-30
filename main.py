import getpass
import os
import re
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your API key here:")

# Setting up an instance of User's Conversation Memory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


prompt_template = ChatPromptTemplate([
    (
        "system",
        """
        You are **OmniScholar**, a highly capable and versatile **AI Research and Knowledge Agent**.
        Your primary goal is to help the user gain deep understanding on *any* topic they inquire about.

        ### Core Responsibilities and Output Quality:
        1.  **Define and Contextualize:** Always begin by providing a clear, precise, and comprehensive **definition** of the main topic.
        2.  **Adapt and Execute:** Rigorously adhere to *all* user-specified commands (e.g., "be brief," "use bullet points," "focus on history," "compare X and Y," "explain it simply"). You *must* integrate these commands into your response structure.
        3.  **Illustrate with Examples:** Always include practical, relevant, and easy-to-understand **examples** or analogies to clarify complex concepts.
        4.  **Use Tools Effectively:** You have access to the *[INSERT_TOOL_NAME_HERE, e.g., web search, database lookup]* tool. Use it *whenever* current, specific, or external information is required to provide the most accurate and precise response.
        5.  **Maintain Conversational Context:** Reference previous parts of the conversation to ensure a cohesive and progressive learning experience for the user.

        **Your tone should be authoritative yet accessible, promoting deep understanding.**
        """
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human",
     """
     My research topic is: **{research_topic}**

     **Specific Commands/Focus:** {specific_commands_or_focus} 

     *If no specific commands are given, assume the user wants a high-level overview with definitions and examples.*
     """
     ),
])


llm = ChatGroq(
    model = "llama-3.1-8b-instant"
)

research_topic = input("Enter the topic that you want to research about:\n ")
specific_commands = input("Enter all the specific commands that you want to execute during your research session:\n ")

chatHistory = memory.load_memory_variables({})["chat_history"]

formatted_prompt = prompt_template.invoke({
    "chat_history":chatHistory,
    "research_topic": research_topic,
    "specific_commands_or_focus": specific_commands
})

'''
def load_and_split_documents(file_path):
    loader =  PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, size= 50)
    splitted_text = text_splitter.split(docs)
    return splitted_text

embeddings_model = HuggingFaceEmbeddings(
    model_name="thenlper/gte-small",
    model_kwargs = {"device":"cpu"}
)
'''

response = llm.invoke(formatted_prompt.messages)

# Splitting the raw response on basis of Heading **...**
sections = re.split(r"\n\*\*(.+?)\*\*\n", response.content)
structured_response = dict()


for i in range(1,len(sections), 2):
    heading=sections[i].strip()
    content = sections[i+1].strip()
    structured_response[heading] = content


print(json.dumps(structured_response,indent=2,ensure_ascii=False))