import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
messages = [
    SystemMessage(content="Translate the following from English into Chinese"),
    HumanMessage(content="Welcome to LLM application development")
]

model = ChatOpenAI(model="deepseek-chat")
stream = model.stream(messages)
for response in stream:
    print(response.content, end="\n")
