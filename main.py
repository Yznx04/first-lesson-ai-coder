import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Translate the following from English into Chinese:"),
    ("user", "{text}")
])

model = ChatOpenAI(model="deepseek-chat")
chain = prompt_template | model
result = chain.invoke({"text": "Welcome to LLM application development!"})
print(result)
