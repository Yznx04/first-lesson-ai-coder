from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv("../.env")

chat_model = ChatOpenAI(model="deepseek-chat")
while True:
    try:
        user_input = input("You:>")
        if user_input.lower() == "exit":
            break
        stream = chat_model.stream([HumanMessage(content=user_input)])
        for chunk in stream:
            print(chunk.content, end='', flush=True)
        print()
    except KeyboardInterrupt:
        break
