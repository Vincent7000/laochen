# ## 固定示例
# 
# 最基本（也是常见）的小样本提示技术是使用固定提示示例。这样您就可以选择一条链条，对其进行评估，并避免担心生产中的额外移动部件。
# 
# 模板的基本组件是： 
# - examples ：要包含在最终提示中的字典示例列表。 
# - example_prompt ：通过其 format_messages 方法将每个示例转换为 1 条或多条消息。一个常见的示例是将每个示例转换为一条人工消息和一条人工智能消息响应，或者一条人工消息后跟一条函数调用消息。

from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

# 组装成少示例的提示模板。

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())

#最后，组装最终的提示并将其与模型一起使用。

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位非常厉害的数学天才。"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
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

chain = final_prompt | chat | output_parser

chain.invoke({"input":"3的平方是多少？"})

chat.invoke(input="3的平方是多少？")

# ## 动态几次提示
# 
# 有时您可能希望根据输入来限制显示哪些示例。为此，您可以将 examples 替换为 example_selector 。其他组件与上面相同！回顾一下，动态几次提示模板将如下所示：
# 
# - example_selector ：负责为给定输入选择少数样本（以及它们返回的顺序）。它们实现了 BaseExampleSelector 接口。一个常见的例子是向量存储支持的 SemanticSimilarityExampleSelector
# 
# - example_prompt ：通过其 format_messages 方法将每个示例转换为 1 条或多条消息。一个常见的示例是将每个示例转换为一条人工消息和一条人工智能消息响应，或者一条人工消息后跟一条函数调用消息。
# 
# 这些可以再次与其他消息和聊天模板组合以组合您的最终提示。

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-large-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "牛对月亮说了什么？", "output": "什么都没有"},
    {
        "input": "给我写一首关于月亮的五言诗",
        "output": "月儿挂枝头，清辉洒人间。 银盘如明镜，照亮夜归人。 思绪随风舞，共赏中秋圆。",
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "对牛弹琴"})

from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

few_shot_prompt.format(input="What's 3+3?")

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位非常厉害的数学天才。"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

print(final_prompt.format(input="3+5是多少？"))

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

chain = final_prompt | chat | output_parser

chain.invoke({"input":"3+5是多少？"})




