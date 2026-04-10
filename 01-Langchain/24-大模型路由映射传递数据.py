# ## 传递数据
# 
# RunnablePassthrough 允许不改变或添加额外的键来传递输入。这通常与 RunnableParallel 结合使用，将数据分配给映射中的新键。
# 
# RunnablePassthrough() 单独调用，将简单地获取输入并将其传递。
# 
# 使用分配 ( RunnablePassthrough.assign(...) ) 调用的 RunnablePassthrough 将获取输入，并将添加传递给分配函数的额外参数。
# 1. RunnableParallel传参数
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

res = runnable.invoke({"num": 1})
print(res)
# 如上所示， passed 键是用 RunnablePassthrough() 调用的，因此它只是传递 {'num': 1} 。
# 在第二行中，我们使用 RunnablePastshrough.assign 和一个将数值乘以 3 的 lambda。在这种情况下， extra 设置为 {'num': 1, 'mult': 3} ，这是原始的添加了 mult 键的值。
# 最后，我们还在映射中使用 modified 设置了第三个键，它使用 lambda 来设置单个值，在 num 上加 1，从而得到 modified 键，其值为 < b2> 。
# ## 运行自定义函数
# 您可以在管道中使用任意函数。
# 请注意，这些函数的所有输入都必须是单个参数。如果您有一个接受多个参数的函数，则应该编写一个接受单个输入并将其解包为多个参数的包装器。


# 2. RunnableLambda传参数
from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("what is {a} + {b}")


chain1 = prompt | model

chain = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")} | RunnableLambda(multiple_length_function),
    }
    | prompt
    | model
)

res = chain.invoke({"foo": "bar", "bar": "gah"})
print(res.content)



# 3. LLM根据提示词，进行分类
# ## 根据输入动态路由逻辑
# 路由允许您创建非确定性链，其中上一步的输出定义下一步。路由有助于提供与 LLMs 交互的结构和一致性。
# 执行路由的方法有两种：
# 1. 有条件地从 RunnableLambda 返回可运行对象（推荐）
# 2. 使用 RunnableBranch 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
        """鉴于下面的用户问题，将其分类为“LangChain”、“OpenAI”或“其他”。

不要用超过一个字来回应。

<question>
{question}
</question>

分类："""
)

chain = (
    prompt
    | model
    | StrOutputParser()
)

res = chain.invoke({"question": "how do I call OpenAI?"})
print(res)

# 4.1 langchain_chain
langchainPrompt = PromptTemplate.from_template(
        """您是 langchain 方面的专家。 
回答问题时始终以“正如老陈告诉我的那样”开头。 
回答以下问题：

问题：{question}
回答："""
)
langchain_chain = langchainPrompt | model

# 4.2 OpenAI_chain
OpenAIPrompt = PromptTemplate.from_template(
        """您是 OpenAI 方面的专家。 
回答问题时始终以“正如奥特曼告诉我的那样”开头。 
回答以下问题：

问题：{question}
回答："""
)
OpenAI_chain = OpenAIPrompt | model

# 4.3 其他general_chain
generalPrompt = PromptTemplate.from_template(
        """ 回答以下问题：

问题：{question}
回答："""
)
general_chain = generalPrompt | model

# 5. 根据LLM判断的分类，进行路由选择chain
def route(info):
    if "OpenAI" in info["topic"]:
        return OpenAI_chain
    elif "LangChain" in info["topic"]:
        return langchain_chain
    else:
        return general_chain

from langchain_core.runnables import RunnableLambda

full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(route)
print(full_chain)

res = full_chain.invoke({"question": "我如何使用OpenAI的模型?"})
print(res.content)



