# ## 文档加载
# 
# 这涵盖了如何加载目录中的所有文档。
# 在底层，默认情况下使用 UnstructedLoader。需要安装依赖
# 
# ```shell
# pip install unstructured
# ```
# python导入方式
# ```
# from langchain_community.document_loaders import DirectoryLoader
# ```
# 
# 我们可以使用 glob 参数来控制加载哪些文件。请注意，此处它不会加载 .rst 文件或 .html 文件。
# ```
# loader = DirectoryLoader('../', glob="**/*.md")
# ```
# 默认情况下不会显示进度条。要显示进度条，请安装 tqdm 库（例如 pip install tqdm ），并将 show_progress 参数设置为 True 。
# ```
# loader = DirectoryLoader('../', glob="**/*.md", show_progress=True)
# docs = loader.load()
# ```

# 1. 向量化几个文件，检索
from langchain_community.document_loaders import TextLoader
import os
print(os.getcwd())
loader = TextLoader("./02-agent/txt/faq-4359.txt",encoding="utf8")
doc = loader.load()
print(doc)

loader1 = TextLoader("./02-agent/txt/faq-7923.txt",encoding="utf8")
doc1 = loader1.load()
print(doc1)

loader2 = TextLoader("./02-agent/txt/yjhx.md",encoding="utf8")
doc2 = loader2.load()
print(doc2)

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)
vectorStoreDB = FAISS.from_documents([doc[0],doc1[0],doc2[0]],embedding=embeddings)
print(vectorStoreDB)

res = vectorStoreDB.similarity_search("回收手机的话，应该怎么操作？")
print(res)

res = vectorStoreDB.similarity_search_with_score("回收手机的话，应该怎么操作？")
print(res)

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

import os
print(os.getcwd())
# 2. 向量化目录，检索
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader("./02-agent/txt", loader_cls=TextLoader)

docs = loader.load()
print(docs)

vectorStoreDB = FAISS.from_documents(docs, embedding=embeddings)
print(vectorStoreDB)

res = vectorStoreDB.similarity_search_with_score("回收手机的话，应该怎么操作？")
print(res)

from langchain_community.document_loaders import UnstructuredHTMLLoader

# loader = UnstructuredHTMLLoader("../02-agent/txt/test.html")
loader = UnstructuredHTMLLoader("./01-Langchain/html/FAQ.html")
# file_path = "/home/vincent/ai课程/老陈-2024-Langchain大模型AI应用与多智能体实战开发/Langchain大模型AI应用实战开发-资料/01-Langchain/html/FAQ.html"

# loader = UnstructuredHTMLLoader(
#     file_path,
#     mode="elements",  # 更快的提取模式
#     strategy="fast"    # 使用快速策略
# )

docs = loader.load()
print(docs)


# 
# 1. **最大边际相关性检索（MMR）**：
# 想象你在一个图书馆里找关于“猫咪”的书籍。MMR就像是一个智能助手，帮你挑选书籍。它会先找到一本最相关的书，比如《猫咪百科全书》。然后，它会寻找第二本书，这本书不仅要与“猫咪”相关，还要与第一本书的内容有所不同，比如可能会找一本关于“猫咪训练”的书。MMR的目标是让你既能找到有用的信息，又能获得不同方面的知识。
# 2. **余弦相似度**：
# 余弦相似度是一种测量两个向量在多维空间中角度的方法。继续用图书馆的例子，你可以把每本书想象成一个向量，它的每个维度代表这本书包含的不同关键词。余弦相似度会计算两本书的向量之间的角度，如果两本书的向量角度接近0度，那么它们在内容上非常相似；如果角度接近90度，那么它们的内容就不那么相似。
# 
# 总结一下：
# - MMR是一个帮你挑选书籍的策略，它力求找到既相关又多样的书籍组合。
# - 余弦相似度是一个测量两本书内容相似度的工具，它通过比较书中的关键词向量来判断。
# 

retriever = vectorStoreDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1}
)

docs = retriever.invoke("请说说回收手机怎么操作？")
print(docs)

# 默认情况下，向量存储检索器使用相似性搜索。如果底层向量存储支持最大边际相关性搜索，您可以将其指定为搜索类型。

#使用向量数据库生成检索器 retriever
retriever = vectorStoreDB.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": 0.3}
)

docs = retriever.invoke("请说说回收手机怎么操作？")
print(docs)


# 执行chain
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

result = setup_and_retrieval.invoke("请说说回收手机怎么操作？")
prompt.invoke(result)



chain = setup_and_retrieval | prompt | model | outputParser
res = chain.invoke("请说说回收手机怎么操作？")
print(res)




