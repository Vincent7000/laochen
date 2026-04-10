# ## PDF
# 
# 可移植文档格式 (PDF)，标准化为 ISO 32000，是 Adobe 于 1992 年开发的一种文件格式，用于以独立于应用程序软件、硬件和操作系统的方式呈现文档，包括文本格式和图像。
# 
# 使用 pypdf 将 PDF 加载到文档数组中，其中每个文档包含页面内容和带有 page 编号的元数据。
# ```SHELL
# pip install pypdf
# ```

from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("./01-Langchain/pdf/2403.04667.pdf")
loader = PyPDFLoader("./01-Langchain/pdf/2403.04732.pdf")
# loader = PyPDFLoader("/home/vincent/_简历/F/许振文的简历.pdf")

pages = loader.load_and_split()

print(pages)

#这种方法的优点是可以使用页码检索文档。
print(pages[0])

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


docs = ""
for item in pages:
    docs += item.page_content
print(docs)

if len(docs) > 2000:
    docs = docs[:2000]  # 截断过长内容
print(docs)


from langchain_core.prompts import ChatPromptTemplate

template ="""
```
{context}
```
总结上面的论文内容
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
outputParser = StrOutputParser()

chain = prompt | model | outputParser
res = chain.invoke({"context": docs})
print(res)

# ## PyPDF 目录
# Load PDFs from directory 从目录加载 PDF

from langchain_community.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("./01-Langchain/pdf/")
docs = loader.load()
print(docs)

if len(docs) > 2000:
    docs = docs[:2000]  # 截断过长内容
print(docs)

from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

vectorStoreDB = FAISS.from_documents(docs, embedding=embeddings)
print(vectorStoreDB)

#使用向量数据库生成检索器
retriever = vectorStoreDB.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={"score_threshold": 0.3}
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


result = setup_and_retrieval.invoke("ChatGPT在法律层面有哪些影响？")
print(result)

chain = setup_and_retrieval | prompt | model | outputParser
res = chain.invoke("ChatGPT在法律层面有哪些影响？")
print(res)






