from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from chapter02.compute_token import tiktoken_counter

load_dotenv("../.env")

chat_model = ChatOpenAI(model="deepseek-chat")
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你现在扮演一个孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’"),
    MessagesPlaceholder(variable_name="messages")
])

trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True
)

with_message_history = RunnableWithMessageHistory(trimmer | prompt_template | chat_model, get_session_history)
config = {"configurable": {"session_id": "yznx"}}

while True:
    try:
        user_input = input("You:>")
        if user_input.lower() == "exit":
            break
        stream = with_message_history.stream({"messages": [HumanMessage(content=user_input)]}, config=config)
        for chunk in stream:
            print(chunk.content, end='', flush=True)
        print()
    except KeyboardInterrupt:
        break
