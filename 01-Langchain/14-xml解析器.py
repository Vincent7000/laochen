# 该输出解析器允许用户以流行的 XML 格式从 LLM 获取结果。
# 
# 请记住，大型语言模型是有漏洞的抽象！您必须使用具有足够容量的 LLM 来生成格式正确的 XML。
# 
# 
# 

from langchain.output_parsers import XMLOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


actor_query = "生成汤姆·汉克斯的电影目录。"
output = model.invoke(
    f"""{actor_query}

响应以xml的结构返回，使用如下xml结构
```
<xml>
<movie>电影1</movie>
<movie>电影2</movie>
<xml>
``` 
"""
)
print(output.content)

parser = XMLOutputParser()
parser.invoke(output)

parser = XMLOutputParser()
# format_instructions =  parser.get_format_instructions()
actor_query = "生成汤姆·汉克斯的电影目录。"
format_instructions = """响应以xml的结构返回，使用如下xml结构
```
<xml>
<movie>电影1</movie>
<movie>电影2</movie>
<xml>
```
"""
prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",
    input_variables=["query"],
    partial_variables={"format_instructions":format_instructions},
)

chain = prompt | model | parser

output = chain.invoke({"query": actor_query})
print(output)

parser = XMLOutputParser()
format_instructions =  parser.get_format_instructions()
format_instructions




