import json
import os
import time
from typing import Optional, Any, Sequence

from dotenv import load_dotenv
from langchain_community.cache import RedisSemanticCache
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv("../.env")

RETURN_VAL_TYPE = Sequence[Generation]


def prompt_key(prompt: str) -> str:
    messages = json.loads(prompt)
    result = [
        "('{}', '{}')".format(data['kwargs']['type'], data['kwargs']['content']) for data in messages
        if 'kwargs' in data and 'type' in data['kwargs'] and 'content' in data['kwargs']
    ]
    return ' '.join(result)


class FixedSemanticCache(BaseCache):

    def __init__(self, cache: BaseCache):
        self.cache = cache

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        key = prompt_key(prompt)
        return self.cache.lookup(key, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        key = prompt_key(prompt)
        return self.cache.update(key, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        return self.cache.clear(**kwargs)


client = OpenAI(
    base_url=os.getenv("QWEN_API_BASE"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
embeddings = DashScopeEmbeddings(
    client=client,
    model="text-embedding-v4"
)

set_llm_cache(FixedSemanticCache(
    RedisSemanticCache(redis_url="redis://localhost:6379", embedding=embeddings)
))

model = ChatOpenAI(model="deepseek-chat")
start_time = time.time()
response = model.invoke("租房窗户靠近马路，不隔音怎么办，给我一些便宜可行的解决方案。")
end_time = time.time()
print(response.content)
print(f"第一次调用耗时：{end_time - start_time}")
start_time = time.time()
response = model.invoke("租房窗户靠近马路，不隔音怎么办，给我一些便宜可行的解决方案。")
end_time = time.time()
print(response.content)
print(f"第二次调用耗时：{end_time - start_time}")
