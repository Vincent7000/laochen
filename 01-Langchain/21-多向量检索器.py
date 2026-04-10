# ## 多向量检索器
# 
# 每个文档存储多个向量通常是有益的。在许多用例中，这是有益的。 LangChain 有一个基础 MultiVectorRetriever ，这使得查询此类设置变得容易。很多复杂性在于如何为每个文档创建多个向量。本笔记本涵盖了创建这些向量和使用 MultiVectorRetriever 的一些常见方法。
# 
# 为每个文档创建多个向量的方法包括：
# - 较小的块：将文档分割成较小的块，然后嵌入这些块（这是 ParentDocumentRetriever）。
# - 摘要：为每个文档创建摘要，将其与文档一起嵌入（或代替文档）
# - 假设性问题：创建每个文档都适合回答的假设性问题，将这些问题与文档一起嵌入（或代替文档）。
# 
# 请注意，这还启用了另一种添加嵌入的方法 - 手动。这很棒，因为您可以显式添加导致文档恢复的问题或查询，从而为您提供更多控制权。


from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

loaders = [
    TextLoader("./02-agent/txt/faq-4359.txt",encoding="utf-8"), # 分期利息
    TextLoader("./02-agent/txt/faq-7923.txt",encoding="utf-8"), # 众测活动
    # TextLoader("./02-agent/txt/项目.txt",encoding="utf-8"),
    # TextLoader("./02-agent/txt/经历.txt",encoding="utf-8"),
]
print(loaders)

docs = []
for loader in loaders:
    docs.extend(loader.load())
print(docs)


from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

# 用于索引子块的向量存储
vectorstore = Chroma(
    collection_name="full_documents", 
    embedding_function=embeddings
)

# 父文档的存储层
store = InMemoryByteStore()
id_key = "doc_id"
# 检索器（空启动）
retriever = MultiVectorRetriever(
# retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

import uuid
doc_ids = [str(uuid.uuid4()) for _ in docs]
print(doc_ids)

from langchain_text_splitters import CharacterTextSplitter
# 用于创建较小块的分割器
child_text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
print(sub_docs)


#使用一个名为 retriever 的对象来向一个向量存储（vectorstore）中添加文档，
#并且使用一个文档存储（ docstore ）来设置文档ID与文档内容之间的映射。
#这两个属性分别用于存储文档的向量化表示和文档的内容。
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# Vectorstore 单独检索小块
res = retriever.vectorstore.similarity_search("众测商品多久发货呢？")[0]
print(res)

# # 摘要总结
# 通常，摘要可能能够更准确地提炼出某个块的内容，从而实现更好的检索。在这里，我们展示如何创建摘要，然后嵌入它们。
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


chain = ({"doc": lambda x: x.page_content}
         | ChatPromptTemplate.from_template("总结下面的文档:\n\n{doc}")
         | model
         | StrOutputParser())

docs = []
for loader in loaders:
    docs.extend(loader.load())
print(docs)

summaries = chain.batch(docs, {"max_concurrency": 5})
print(summaries)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=embeddings)

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
# retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
print(summary_docs)

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

sub_docs = retriever.vectorstore.similarity_search("众测活动是否有参与限制？")
print(sub_docs)

# ## 假设查询
# LLM 还可用于生成针对特定文档可能提出的假设问题列表。然后可以嵌入这些问题
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
print(sub_docs)

# 根据上面的文档，生成3个相关问题和回答。响应以json列表的结构返回。返回的结构参考如下
from langchain_core.output_parsers import JsonOutputParser
promptStr = '''
```
{doc}
```
根据上面的文档，生成3个相关问题和回答。响应以json列表的结构返回。返回的结构参考如下
请严格按照以下 JSON 格式输出，不要添加任何解释、思考过程、Markdown 格式（如 ```json）或额外的文字。

```
[
{{"question":"问题1","answer":"回答1"}},
{{"question":"问题2","answer":"回答2"}},
{{"question":"问题3","answer":"回答3"}}
]
```
'''

prompt = ChatPromptTemplate.from_template(promptStr)

chain = (
    {"doc": lambda x: x.page_content}
    | prompt
    | model
    | JsonOutputParser()
)

hypothetical_questions = chain.batch(sub_docs, {"max_concurrency": 5})

print(hypothetical_questions)

documents = []
for item in hypothetical_questions:
    for obj in item:
        content = "问：{}\n答：{}".format(obj['question'],obj['answer'])
        documents.append(Document(page_content=content))
print(documents)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="Question", 
    embedding_function=embeddings,
    persist_directory="./vector_store"
    )
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
# retriever = ParentDocumentRetriever(    
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

retriever.vectorstore.add_documents(documents)


res = retriever.vectorstore.similarity_search("众测商品多久发货呢？")[0]
print(res)




