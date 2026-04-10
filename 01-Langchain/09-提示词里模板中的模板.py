# ## MessagePromptTemplate
# 
# LangChain提供了不同类型的 MessagePromptTemplate 。最常用的是 AIMessagePromptTemplate 、 SystemMessagePromptTemplate 和 HumanMessagePromptTemplate ，它们分别创建 AI 消息、系统消息和人工消息。
# 
# 但是，如果聊天模型支持使用任意角色获取聊天消息，则可以使用 ChatMessagePromptTemplate ，它允许用户指定角色名称。

# 

from langchain.prompts import ChatMessagePromptTemplate

prompt = "愿{subject}与你同在"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="Jedi", template=prompt
)
chat_message_prompt.format(subject="上帝")

# LangChain 还提供了 MessagesPlaceholder ，它使您可以完全控制格式化期间要呈现的消息。当您不确定消息提示模板应使用什么角色或希望在格式化期间插入消息列表时，这会很有用。

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

human_prompt = "用 {word_count} 字总结我们迄今为止的对话。"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

from langchain_core.messages import AIMessage, HumanMessage

human_message = HumanMessage(content="学习编程最好的方法是什么?")
ai_message = AIMessage(
    content="""\
1. 选择编程语言：决定您想要学习的编程语言。

2. 从基础开始：熟悉变量、数据类型和控制结构等基本编程概念。

3. 练习、练习、再练习：学习编程的最好方法是通过实践经验\
"""
)

chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="10"
).to_messages()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
# openai_api_base = "http://127.0.0.1:1234/v1"
openai_api_base = "http://localhost:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.7,
)

output_parser = StrOutputParser()

chain = chat_prompt | chat | output_parser

chain.invoke({"word_count":"10","conversation":[human_message, ai_message]})

# ## 部分提示模板
# 
# 与其他方法一样，“部分”提示模板是有意义的 - 例如传入所需值的子集，以创建一个仅需要剩余值子集的新提示模板。
# 
# LangChain 通过两种方式支持这一点： 
# 1. 使用字符串值进行部分格式化。 
# 2. 使用返回字符串值的函数进行部分格式化。
# 
# 这两种不同的方式支持不同的用例。在下面的示例中，我们将回顾这两个用例的动机以及如何在 LangChain 中实现这一点。
# 
# 
# ### 部分带字符串
# 
# 想要部分提示模板的一个常见用例是，如果您先于其他变量获取某些变量。例如，假设您有一个提示模板需要两个变量 foo 和 baz 。如果您在链的早期获得 foo 值，但稍后获得 baz 值，那么等到两个变量都位于同一位置才能将它们传递给提示模板。相反，您可以使用 foo 值部分化提示模板，然后传递部分化的提示模板并直接使用它。下面是执行此操作的示例：
# 
# 

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{foo}{bar}")
partial_prompt = prompt.partial(foo="foo")
print(partial_prompt.format(bar="baz"))

#您还可以仅使用部分变量初始化提示。

prompt = PromptTemplate(
    template="{foo}{bar}", input_variables=["bar"], partial_variables={"foo": "foo"}
)
print(prompt.format(bar="baz"))

# ## 部分函数
# 
# 另一个常见用途是对函数进行部分处理。这样做的用例是当您有一个变量时，您知道您总是希望以通用方式获取该变量。一个典型的例子是日期或时间。想象一下，您有一个提示，您总是希望获得当前日期。您无法在提示中对其进行硬编码，并且将其与其他输入变量一起传递有点烦人。在这种情况下，能够使用始终返回当前日期的函数来部分提示是非常方便的。

from datetime import datetime


def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y")

prompt = PromptTemplate(
    template="给我讲一个关于 {date} 天的 {adjective} 故事",
    input_variables=["adjective", "date"],
)
partial_prompt = prompt.partial(date=_get_datetime())
print(partial_prompt.format(adjective="有趣"))

prompt = PromptTemplate(
    template="给我讲一个关于 {date} 天的 {adjective} 故事",
    input_variables=["adjective"],
    partial_variables={"date": _get_datetime},
)
print(prompt.format(adjective="funny"))




