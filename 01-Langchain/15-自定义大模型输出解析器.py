# ## 自定义输出解析器
# 
# 在某些情况下，您可能希望实现自定义解析器以将模型输出构造为自定义格式。
# 
# 有两种方法可以实现自定义解析器：
# - 在 LCEL 中使用 RunnableLambda 或 RunnableGenerator – 我们强烈建议大多数用例使用此方法
# - 通过从基类之一继承进行解析——这是困难方法
# 
# 这两种方法之间的差异大多是表面的，主要在于触发哪些回调（例如， on_chain_start 与 on_parser_start ）
# 
# ### 可运行的 Lambda 和生成器
# 
# 推荐的解析方法是使用可运行的 lambda 和可运行的生成器！
# 
# 在这里，我们将进行一个简单的解析，反转模型输出的大小写。
# 
# 例如，如果模型输出：“Meow”，解析器将生成“mEOW”。
# 

from typing import Iterable
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


def parse(ai_message: AIMessage) -> str:
    """Parse the AI message."""
    return ai_message.content.swapcase()


chain = model | parse
chain.invoke("Hello")

for chunk in chain.stream("tell me a story"):
    print(chunk, end="")

from langchain_core.runnables import RunnableGenerator


def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.content.swapcase()


streaming_parse = RunnableGenerator(streaming_parse)

chain = model | streaming_parse
for chunk in chain.stream("tell me a story"):
    print(chunk, end="")

# 




