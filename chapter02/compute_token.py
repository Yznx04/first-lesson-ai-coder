from typing import List

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage


def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 1
    tokens_per_name = 1
    for msg in messages:
        match msg:
            case HumanMessage():
                role = "user"
            case AIMessage():
                role = "assistant"
            case ToolMessage():
                role = "tool"
            case SystemMessage(msg):
                role = "system"
            case _:
                raise ValueError(f"Unsupported message type: {msg.__class__}")
        print(f"role: {role}")
        num_tokens += (
                tokens_per_message + str_token_counter(role) + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens
