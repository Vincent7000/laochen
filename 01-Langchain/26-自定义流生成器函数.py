# ## 流自定义生成器函数
# 您可以在 LCEL 管道中使用生成器函数（即使用 yield 关键字且行为类似于迭代器的函数）。
# 这些生成器的签名应该是 Iterator[Input] -> Iterator[Output] 。或者对于异步生成器： AsyncIterator[Input] -> AsyncIterator[Output] 。
# 这些对于： - 实现自定义输出解析器 - 修改上一步的输出，同时保留流功能
# 让我们为逗号分隔列表实现一个自定义输出解析器。

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)

from typing import Iterator, List

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_template(
    "响应以CSV的格式返回中文列表，不要返回其他内容。请输出与{transportation}类似的交通工具"
)

str_chain = prompt | model | StrOutputParser()

res = str_chain.invoke({"transportation":"飞机"})
print(res)

for chunk in str_chain.stream({"transportation":"飞机"}):
    print(chunk, end="", flush=True)

# 这是一个自定义解析器，用于拆分 llm 令牌的迭代器
# 放入以逗号分隔的字符串列表中
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    # 保留部分输入，直到得到逗号
    buffer = ""
    for chunk in input:
        # 将当前块添加到缓冲区
        buffer += chunk
        # 当缓冲区中有逗号时
        while "," in buffer:
            # 以逗号分割缓冲区
            comma_index = buffer.index(",")
            # 产生逗号之前的所有内容
            yield [buffer[:comma_index].strip()]
            # 保存其余部分以供下一次迭代使用
            buffer = buffer[comma_index + 1 :]
    # 产生最后一个块
    yield [buffer.strip()]

list_chain = str_chain | split_into_list
for chunk in list_chain.stream({"transportation":"飞机"}):
    print(chunk, end="", flush=True)

# ## 异步版本

from typing import AsyncIterator


async def asplit_into_list(
    input: AsyncIterator[str],
) -> AsyncIterator[List[str]]:  # async def
    buffer = ""
    async for (
        chunk
    ) in input:  # `input` 是一个 `async_generator` 对象，所以使用 `async for`
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1 :]
    yield [buffer.strip()]


list_chain = str_chain | asplit_into_list

# async for chunk in list_chain.astream({"transportation":"飞机"}):
#     print(chunk, flush=True)

# await list_chain.ainvoke({"transportation":"飞机"})




