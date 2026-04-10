# ## CSV解析器
# 当您想要返回以逗号分隔的项目列表时，可以使用此输出解析器。


from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)

chat.invoke("请问2只兔子有多少条腿？")

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()


format_instructions = "您的响应应该是csv格式的逗号分隔值的列表，例如：`内容1, 内容2, 内容3`"
prompt = PromptTemplate(
    template="{format_instructions}\n请列出五个 {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)



chain = prompt | chat | output_parser

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

res = chain.invoke({"subject": "冰淇淋口味"})
print(res)

res = prompt.invoke({"subject": "冰淇淋口味"})
print(res)


# ## Enum parser 枚举解析器

from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum

class Colors(Enum):
    RED = "红色"
    BROWN = "棕色"
    BLACK = "黑色"
    WHITE = "白色"
    YELLOW = "黄色"
    

parser = EnumOutputParser(enum=Colors)

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

promptTemplate = PromptTemplate.from_template(
    """{person}的皮肤主要是什么颜色？

{instructions}"""
)

# instructions=parser.get_format_instructions()
instructions = "响应的结果请选择以下选项之一：红色、棕色、黑色、白色、黄色。不要有其他的内容"
prompt = promptTemplate.partial(instructions=instructions)
chain = prompt | chat | parser

instructions=parser.get_format_instructions()
print(instructions)

res = chain.invoke({"person": "亚洲人"})
print(res)


# ## 日期时间解析器
# 此 OutputParser 可用于将 LLM 输出解析为日期时间格式。

from langchain.output_parsers import DatetimeOutputParser
output_parser = DatetimeOutputParser()
template = """回答用户的问题:

{question}

{format_instructions}"""


format_instructions=output_parser.get_format_instructions()

format_instructions='''响应的格式用日期时间字符串:“%Y-%m-%dT%H:%M:%S.%fZ”。

示例: 1898-05-31T06:59:40.248940Z, 1808- 10-20T01:56:09.167633Z、0226-10-17T06:18:24.192024Z

仅返回此字符串，没有其他单词！'''
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions":format_instructions },
)

print(format_instructions)

chain = prompt | chat | output_parser

output = chain.invoke({"question": "比特币是什么时候创立的？"})
print(output)



