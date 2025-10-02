import getpass
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from document_store import load_vectorstore, ingest_documents, search_documents

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

PERSIST_DIR="vector_store"
retriever = load_vectorstore(PERSIST_DIR,device="cpu")

while(True):
    choice = int(input("Enter the number to continue or to quit:\n1.Continue\n2.Exit\n3.Print Conversation Memory\n4.Upload file paths"))

    if(choice == 2):
        print("\nExiting the Loop\n")
        break

    elif (choice==3):
        print(memory.load_memory_variables({})["chat_history"])
        continue

    elif (choice==4):
        paths_input = input("Enter file paths separated by comma (eg. ./docs/a.pdf, ./docs/b.pdf, ./textFiles/notes.txt): \n")
        paths = [p.strip() for p in paths_input.split(",") if p.strip()]

        try:
            index =ingest_documents(paths,persist_dir=PERSIST_DIR, device="cpu")
            retriever = index.as_retriever(search_kwargs={"k":4})
            print("Ingested and saved index to : ", PERSIST_DIR)

        except Exception as e:
            print("Ingest failed : ", e)

        continue
 


    research_topic = input("Enter the topic that you want to research about:\n ")
    specific_commands = input("Enter all the specific commands that you want to execute during your research session:\n ")

    chatHistory = memory.load_memory_variables({})["chat_history"]


    # Attatching the user uploaded document context to prompt 'specific commands' section.
    context_docs=""
    if (not retriever == None):
        docs = search_documents(retriever,research_topic,k=4)
        context_docs = "\n\n---\n\n".join([d.page_content for d in docs[:4]])

        if(context_docs):
            specific_commands = f"DOCUMENT_CONTEXT_START\n{context_docs}\nDOCUMENT_CONTEXT_END\n\n{specific_commands}"

    formatted_prompt = prompt_template.invoke({
        "chat_history": chatHistory,
        "research_topic": research_topic,
        "specific_commands_or_focus": specific_commands
    })

    response = llm.invoke(formatted_prompt.messages)

    # Splitting the raw response on basis of Heading **...**
    sections = re.split(r"\n\*\*(.+?)\*\*\n", response.content)
    structured_response = dict()


    for i in range(1,len(sections), 2):
        heading=sections[i].strip()
        content = sections[i+1].strip()
        structured_response[heading] = content



    print("\n...Saving to Conversation Memory\n")
    memory.save_context({
        "input": f"My research topic is :{research_topic}\nSpecific commands: {specific_commands}"},
        {"output":response.content}
    )

    print("\nSaved to Memory Successfully !\n")
    print(structured_response)

    stored_messages = memory.load_memory_variables({})

    print("\n---Stored Conversation Memory ---\n")
