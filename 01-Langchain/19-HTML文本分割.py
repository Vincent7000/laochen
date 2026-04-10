# ## Text Splitters 文本分割器
# 
# 加载文档后，您通常会想要对其进行转换以更好地适合您的应用程序。最简单的例子是，您可能希望将长文档分割成更小的块，以适合模型的上下文窗口。 LangChain 有许多内置的文档转换器，可以轻松地拆分、组合、过滤和以其他方式操作文档。
# 
# 当您想要处理长文本时，有必要将该文本分割成块。这听起来很简单，但这里存在很多潜在的复杂性。理想情况下，您希望将语义相关的文本片段保留在一起。 “语义相关”的含义可能取决于文本的类型。本笔记本展示了实现此目的的几种方法。
# 
# 在较高层面上，文本分割器的工作原理如下：
# - 将文本分成小的、具有语义意义的块（通常是句子）。
# - 开始将这些小块组合成一个更大的块，直到达到一定的大小（通过某些函数测量）。
# - 一旦达到该大小，请将该块设为自己的文本片段，然后开始创建具有一些重叠的新文本块（以保持块之间的上下文）。
# 
# ### HTMLHeaderTextSplitter
# 
# “MarkdownHeaderTextSplitter”、“HTMLHeaderTextSplitter”是一个“结构感知”分块器，它在元素级别拆分文本，并为每个与任何给定块“相关”的标题添加元数据。它可以逐个元素返回块，或者将元素与相同的元数据组合起来，目的是 (a) 保持相关文本在语义上（或多或少）分组，以及 (b) 保留文档结构中编码的上下文丰富的信息。它可以与其他文本分割器一起使用，作为分块管道的一部分。

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./01-Langchain/html/Animation-system.html", encoding="utf8")
doc = loader.load()
print(doc)

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# pip install -qU langchain-text-splitters

from langchain_text_splitters import HTMLHeaderTextSplitter
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(doc[0].page_content)
print(html_header_splits)

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

vectorStoreDB = FAISS.from_documents(html_header_splits,embedding=embeddings)
print(vectorStoreDB)

retriever = vectorStoreDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}
)

# docs = retriever.get_relevant_documents("请说说如何控制动画的播放和暂停")
docs = retriever.invoke("请说说如何控制动画的播放和暂停")
print(docs)



from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
template ="""
只根据以下文档回答问题：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)
outputParser = StrOutputParser()
setup_and_retrieval = RunnableParallel(
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }
)

result = setup_and_retrieval.invoke("请说说如何控制动画的播放和暂停？")
print(result)

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


chain = setup_and_retrieval | prompt | model | outputParser
res = chain.invoke("请说说如何控制动画的播放和暂停？")
print(res)



