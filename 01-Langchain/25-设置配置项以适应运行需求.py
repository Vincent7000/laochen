# ## 绑定运行时参数

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "用代数符号写出下面的方程，然后求解。 使用格式\n\n方程:...\n解决方案:...\n\n",
        ),
        ("human", "{equation_statement}"),
    ]
)

chain = (
    {"equation_statement": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

res = chain.invoke("x的3次方加7等于34")
print(res)


# ## 在运行时配置链的内部结构
# 通常，您可能想要尝试多种不同的做事方式，甚至向最终用户展示多种不同的做事方式。为了使这种体验尽可能简单，我们定义了两种方法。
# 首先是 configurable_fields 方法。这允许您配置可运行的特定字段。
# 其次，一个 configurable_alternatives 方法。使用此方法，您可以列出可以在运行时设置的任何特定可运行对象的替代方案。

from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

model = model.configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

res = model.invoke("生成一个随机的整数，不要回答其他任何内容")
print(res.content)

res = model.invoke("生成一个随机的整数，不要回答其他任何内容")
print(res.content)

res = model.with_config(configurable={"llm_temperature": 0.9}).invoke("生成一个随机的整数，不要回答其他任何内容")
print(res.content)

# ## `@chain` decorator
# 使用“@chain”装饰器创建一个可运行程序,您还可以通过添加 @chain 装饰器将任意函数变成链。这在功能上相当于包装在 RunnableLambda 中。
# 通过正确跟踪您的链，这将具有提高可观察性的好处。对该函数内的可运行对象的任何调用都将被跟踪为嵌套子函数。
# 它还允许您像任何其他可运行程序一样使用它，将其组合成链等。

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

prompt1 = ChatPromptTemplate.from_template("给我讲一个关于 {topic} 的故事")
prompt2 = ChatPromptTemplate.from_template("{story}\n\n对上面这个故事进行修改，让故事变得更加口语化和幽默有趣")

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.8,
)

@chain
def custom_chain(text):
    prompt_val1 = prompt1.invoke({"topic":text })
    output1 = model.invoke(prompt_val1)
    parsed_output1 = StrOutputParser().invoke(output1)
    
    chain2 = prompt2 | model | StrOutputParser()
    return chain2.invoke({"story": parsed_output1})

@chain
def custom_chain2(text):
    chain1 = prompt1 | model | StrOutputParser()
    parsed_output1 = chain1.invoke({"topic":text })
    
    chain2 = prompt2 | model | StrOutputParser()
    return chain2.invoke({"story": parsed_output1})

from langchain_core.runnables import RunnablePassthrough

@chain
def custom_chain3(text):
    chain = prompt1 | model | StrOutputParser()| {"story": RunnablePassthrough()} | prompt2 | model | StrOutputParser()
    return chain.invoke({"topic":text })

r1 = custom_chain.invoke("有志者事竟成")
print(r1)

r2 = custom_chain2.invoke("有志者事竟成")
print(r2)

r3 = custom_chain3.invoke("有志者事竟成")
print(r3)




