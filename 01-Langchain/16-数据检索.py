# ## 数据检索
# 用户输入只是问题，答案需要从已知的数据库或者文档中获取。因此，我们需要使用检索器获取上下文，并通过“question”键下的用户输入。
# 
# ### RAG
# 检索增强生成（Retrieval-augmented Generation，RAG），是当下最热门的大模型前沿技术之一。如果将“微调（finetune）”理解成大模型内化吸收知识的过程，那么RAG就相当于给大模型装上了“知识外挂”，基础大模型不用再训练即可随时调用特定领域知识。

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
# embeddings_path = "/home/vincent/.lmstudio/models/bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"

embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

vectorstore = FAISS.from_texts(
    ["小明在华为工作","熊喜欢吃蜂蜜"],
    embedding=embeddings
)
print(vectorstore)



#使用向量数据库生成检索器
retriever = vectorstore.as_retriever()

res = retriever.invoke("熊喜欢吃什么？")
print(res)


from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


from langchain_core.prompts import ChatPromptTemplate

template ="""
只根据以下文档回答问题：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

outputParser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }
)

chain = setup_and_retrieval | prompt | model | outputParser
res = chain.invoke("小明在哪里工作？")
print(res)

#完整链式
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

res = chain.invoke("小明在哪里工作？")
print(res)

from operator import itemgetter

template ="""
只根据以下文档回答问题：
{context}

问题：{question}
回答问题请加上称呼"{name}"。
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "name": itemgetter("name"),
    }
    | prompt
    | model
    | StrOutputParser()
)

res = chain.invoke({"question":"小明在哪里工作？","name":"主人"})
print(res)

import os
print(os.getcwd())

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader("/home/vincent/ai课程/老陈-2024-Langchain大模型AI应用与多智能体实战开发/Langchain大模型AI应用实战开发-资料/02-agent/txt",
                         loader_cls=TextLoader)

docs = loader.load()
print(docs)

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader(
    "/home/vincent/ai课程/老陈-2024-Langchain大模型AI应用与多智能体实战开发/Langchain大模型AI应用实战开发-资料/02-agent/txt",
    # glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True
)
docs = loader.load()
print(f"成功加载 {len(docs)} 个文档")

print(docs)




