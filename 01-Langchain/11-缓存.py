# ## Caching 缓存
# 
# LangChain为LLMs提供了可选的缓存层。这很有用，原因有两个：
# 
# - 如果您经常多次请求相同的完成，它可以通过减少您对 LLM 提供程序进行的 API 调用次数来节省资金。
# - 它可以通过减少您对 LLM 提供程序进行的 API 调用次数来加速您的应用程序。

from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.7,
)

res = chat.invoke("请问2只兔子有多少条腿？")
print(res)

from langchain.globals import set_llm_cache
# from langchain.cache import InMemoryCache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())




from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
output_parser = StrOutputParser()

chain = prompt | chat | output_parser

res = chain.invoke({"topic": "康师傅绿茶"})
print(res)


res = chain.invoke({"topic": "康师傅绿茶"})
print(res)

# ## SQLite 缓存

# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="./db/langchain.db"))


res = chain.invoke({"topic": "旺仔小馒头"})
print(res)

res = chain.invoke({"topic": "旺仔小馒头"})
print(res)





