# 管道提示词可以将多个提示组合在一起。当您想要重复使用部分提示时，这会很有用。这可以通过 PipelinePrompt 来完成。
# 
# PipelinePrompt 由两个主要部分组成：
# - 最终提示：返回的最终提示
# - 管道提示：元组列表，由字符串名称和提示模板组成。每个提示模板将被格式化，然后作为具有相同名称的变量传递到未来的提示模板。

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate


# 1.
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

# 2.1
introduction_template = """你正在冒充{person}。"""
introduction_prompt = PromptTemplate.from_template(introduction_template)

# 2.2
example_template = """
下面是一个交互示例：

Q：{example_q}
A：{example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

# 2.3
start_template = """现在正式开始！

Q：{input}
A："""
start_prompt = PromptTemplate.from_template(start_template)

# -> 2. 
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]

# 3. 
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

print(pipeline_prompt.input_variables)

print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="你最喜欢什么车？",
        example_a="Tesla",
        input="您最喜欢的社交媒体网站是什么?",
    )
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.7,
)

output_parser = StrOutputParser()

chain = pipeline_prompt | chat | output_parser

res = chain.invoke({
    "input":"您最喜欢的社交媒体网站是什么",
    "person":"Elon Musk",
    "example_q":"你最喜欢什么车？",
    "example_a":"Tesla",
})

print(res)


