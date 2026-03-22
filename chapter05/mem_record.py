from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from mem0 import Memory

load_dotenv("../.env")
config = {
    "version": "v1.1",
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": "deepseek-chat",
            "temperature": 0,
            "max_tokens": 1500
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "deepseek-chat",
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0db",
            "path": "mem0db"
        }
    },
    "history_db_path": "history.db"
}

mem0 = Memory.from_config(config)

llm = ChatOpenAI(model="deepseek-chat")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你现在扮演一个孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’.
    利用上下文进行个性化回复，并记住用户的偏好和以往的交互行为。
    上下文: {context}
    """),
    ("user", "{input}")
])

chain = prompt_template | llm


def retrieve_content(query: str, user_id: str) -> str:
    memories = mem0.search(query, user_id=user_id)
    return " ".join(mem["memory"] for mem in memories["result"])


def save_interaction(user_id: str, user_input: str, assistant_response: str):
    interaction = [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    mem0.add(interaction, user_id=user_id)


def invoke(user_input: str, user_id: str) -> str:
    context = retrieve_content(user_input, user_id)
    response = chain.invoke({
        "content": context,
        "input": user_input
    })
    content = response.content
    save_interaction(user_id, user_input, content)
    return content


user_id = "yznx"

while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    response = invoke(user_input, user_id)
    print(response)
