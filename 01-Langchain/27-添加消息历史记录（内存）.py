# ## 添加消息历史记录（内存）
# RunnableWithMessageHistory 允许我们将消息历史记录添加到某些类型的链中。它包装另一个 Runnable 并管理它的聊天消息历史记录。
# 具体来说，它可用于任何将以下之一作为输入的 Runnable
# - 一个字典，其键采用 BaseMessage 序列
# - 一个字典，其键将最新消息作为 BaseMessage 的字符串或序列，以及一个单独的键，将历史消息作为字符串或序列
# 并作为输出之一返回
# - 可以视为 AIMessage 内容的字符串
# - 一个字典，其键包含 BaseMessage 序列
# 让我们看一些示例，看看它是如何工作的。首先我们构造一个可运行的程序（这里接受一个字典作为输入并返回一条消息作为输出）：


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位擅长{ability}的助手。 用 20 个字或更少的字数进行简洁概要的回复",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model

# 为了管理消息历史记录，我们需要： 
# 1. 这个可运行程序； 
# 2. 返回 BaseChatMessageHistory 实例的可调用函数。
# 
# 查看内存集成页面，了解使用 Redis 和其他提供程序实现聊天消息历史记录。在这里，我们演示如何使用内存中 ChatMessageHistory 以及使用 RedisChatMessageHistory 进行更持久的存储。

# ## In-memory 内存中
# 
# 下面我们展示了一个简单的示例，其中聊天历史记录位于内存中，在本例中通过全局 Python 字典。
# 
# 我们构造一个可调用的 get_session_history ，它引用此字典以返回 ChatMessageHistory 的实例。可以通过在运行时将配置传递给 RunnableWithMessageHistory 来指定可调用的参数。
# 默认情况下，配置参数应为单个字符串 session_id 。这可以通过 history_factory_config kwarg 进行调整。
# 
# 使用单参数默认值：

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 请注意，我们指定了 input_messages_key （被视为最新输入消息的键）和 history_messages_key （添加历史消息的键）。
# 
# 当调用这个新的可运行时，我们通过配置参数指定相应的聊天历史记录：

res = with_message_history.invoke(
    {"ability": "数学", "input": "余弦是什么意思？"},
    config={"configurable": {"session_id": "abc123"}},
)
print(res.content)

res = with_message_history.invoke(
    {"ability": "math", "input": "能否再说一遍？"},
    config={"configurable": {"session_id": "abc123"}},
)
print(res.content)

# New session_id --> does not remember.
res = with_message_history.invoke(
    {"ability": "math", "input": "能否再说一遍？"},
    config={"configurable": {"session_id": "def234"}},
)
print(res.content)

# 我们可以通过将 ConfigurableFieldSpec 对象列表传递给 history_factory_config 参数来自定义用于跟踪消息历史记录的配置参数。下面，我们使用两个参数： user_id 和 conversation_id 。

from langchain_core.runnables import ConfigurableFieldSpec

store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符。",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="对话的唯一标识符。",
            default="",
            is_shared=True,
        ),
    ],
)

res = with_message_history.invoke(
    {"ability": "math", "input": "你好"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)
print(res.content)

# ## 具有不同签名的可运行实例的示例
# 上面的可运行程序接受一个字典作为输入并返回一个 BaseMessage。下面我们展示了一些替代方案。

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({"output_message":model})


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    output_messages_key="output_message",
)

res = with_message_history.invoke(
    [HumanMessage(content="什么是二元一次方程")],
    config={"configurable": {"session_id": "baz"}},
)
print(res['output_message'].content)

res = with_message_history.invoke(
    [HumanMessage(content="什么是未知数")],
    config={"configurable": {"session_id": "baz"}},
)
print(res['output_message'].content)

RunnableWithMessageHistory(
    model,
    get_session_history,
)

from operator import itemgetter

RunnableWithMessageHistory(
    itemgetter("input_messages") | model,
    get_session_history,
    input_messages_key="input_messages",
)

# ## 持久存储
# 
# 在许多情况下，最好保留对话历史记录。 RunnableWithMessageHistory 不知道 get_session_history 可调用对象如何检索其聊天消息历史记录。有关使用本地文件系统的示例，请参阅此处。下面我们演示如何使用 Redis。查看内存集成页面，了解使用其他提供程序实现聊天消息历史记录。
# 
# 如果尚未安装 Redis，我们需要安装它：
# 
# pip install --upgrade --quiet redis
# 
# 如果我们没有可连接的现有 Redis 部署，请启动本地 Redis Stack 服务器：
# 
# docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
# 
# REDIS_URL = "redis://localhost:6379/0"
# 
# 更新消息历史记录实现只需要我们定义一个新的可调用对象，这次返回 RedisChatMessageHistory 的实例：
# 
# 
# 
# 
# 

from langchain_community.chat_message_histories import RedisChatMessageHistory
import os
# 尝试从系统环境中获取 REDIS_URL，如果获取不到则使用默认值
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

res = with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "foobar"}},
)
print(res.content)

res = with_message_history.invoke(
    {"ability": "math", "input": "What's its inverse"},
    config={"configurable": {"session_id": "foobar"}},
)
print(res.content)


