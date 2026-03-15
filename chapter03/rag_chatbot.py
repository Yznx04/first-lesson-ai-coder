import os
from operator import itemgetter
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from openai import OpenAI

from chapter02.compute_token import str_token_counter

load_dotenv("../.env")
client = OpenAI(
    base_url=os.getenv("QWEN_API_BASE"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
embeddings = DashScopeEmbeddings(
    client=client,
    model="text-embedding-v4"
)
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=embeddings,
    persist_directory="vectordb"
)
retriever = vectorstore.as_retriever(search_type="similarity")


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (tokens_per_message + str_token_counter(role) + str_token_counter(msg.content))
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
        return num_tokens


trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True
)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


model = ChatOpenAI(model="deepseek-chat")
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an assistant for question-answering tasks. 
     Use the following pieces of retrieved context to answer the question. 
     If you don't know the answer, just say that you don't know. 
     Use three sentences maximum and keep the answer concise. Context: {context}""",),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


context = itemgetter("question") | retriever | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | trimmer | model
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "yznx"}}
while True:
    try:
        user_input = input("You:>")
        if user_input.lower() == "exit":
            break
        if user_input.strip() == "":
            continue
        stream = with_message_history.stream({"question": user_input}, config=config)
        for chunk in stream:
            print(chunk.content, end='', flush=True)
        print()
    except KeyboardInterrupt:
        break
