import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Work(BaseModel):
    title: str = Field(..., description="Title of the work")
    description: str = Field(..., description="Description of the work")


os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
parser = JsonOutputParser(pydantic_object=Work)
prompt_template = PromptTemplate(
    template="列举3部{author}的作品。\n{format_instructions}",
    input_variables=["author"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatOpenAI(model="deepseek-chat")
chain = prompt_template | model | parser
result = chain.invoke({"author": "刘震云"})
print(result)
