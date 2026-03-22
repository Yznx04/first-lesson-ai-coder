import time

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv("../.env")

set_llm_cache(InMemoryCache())
model = ChatOpenAI(model="deepseek-chat")
start_time = time.time()
response = model.invoke("租房窗户靠近马路，不隔音怎么办，给我一些便宜可行的解决方案。")
end_time = time.time()
print(response.content)
print(f"第一次调用耗时：{end_time - start_time}")
start_time = time.time()
response = model.invoke("能不能给我一些关于租房窗户靠近马路不隔音这个问题的解决方案。")
end_time = time.time()
print(response.content)
print(f"第二次调用耗时：{end_time - start_time}")
