import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv("../.env")

loader = TextLoader("introduction.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
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
vectorstore.add_documents(splits)
documents = vectorstore.similarity_search("专栏的作者是谁？")
print(documents)
