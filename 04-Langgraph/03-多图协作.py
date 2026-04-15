# %% [markdown]
# # 多图协作
# 
# 编译 langgraph 实例会将其转换为常规的可运行的 langchain。 这可以用作任何其他图中的节点。
# 
# 创建子图可以让您构建诸如多代理团队之类的东西，其中每个团队都可以跟踪自己单独的状态。
# 
# 下面是一个简单的（有点做作的）图示例，其中一个节点本身就是一个图。 该子图将包含一个简单的无工具“代理”，它生成响应，然后循环自我不断改进

# %%
# %pip install -U langgraph langchain_anthropic

# %%
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utils import llm


# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LANGCHAIN_API_KEY")
# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("ANTHROPIC_API_KEY")

# # %%
# from langchain_openai import ChatOpenAI, OpenAI

# openai_api_key = "EMPTY"
# openai_api_base = "http://127.0.0.1:1234/v1"
# # llm = ChatOpenAI(
# #     openai_api_key=openai_api_key,
# #     openai_api_base=openai_api_base,
# #     temperature=0.3,
# #     model_kwargs = {
# #         "frequency_penalty":0.9,
# #         "presence_penalty":0.9
# #     }
# # )
# llm = ChatOpenAI(
#     openai_api_key=openai_api_key,
#     openai_api_base=openai_api_base,
#     temperature=0.3,
# )

# %% [markdown]
# ## Subgraph
# 
# 我们的玩具子图将是一个简单的循环，它生成一个笑话，然后自我批评。

# %%
import operator
from typing import Annotated, List, TypedDict

# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

# llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你就是那个爱开玩笑的人。 用一个笑话来回应，这是有史以来最好的笑话。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ],
)


critic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """{message}
-------
请提出对这个笑话的改进建议，使其成为有史以来最好的笑话。"""),
    ],
)


def update(out):
    return {"messages": [("assistant",out.content)]}


def replace_role(out):
    print("replace_role--------")
    print(out)
    print("replace_role_end--------")
    return {"messages": [HumanMessage(out.content)]}

def critiqueFn(state):
    print("state---------------------")
    print(state)
    message = state["messages"][-1]
    print(message)
    print("state end---------------------")
    return {"message":message[1]}

# %%
## 构建子图


class SubGraphState(TypedDict):
    messages: Annotated[List, operator.add]


builder = StateGraph(SubGraphState)
builder.add_node("tell_joke", prompt | llm | update)
builder.add_node("critique", critiqueFn |critic_prompt | llm | replace_role)


def route(state):
    return END if len(state["messages"]) >= 3 else "critique"


builder.add_conditional_edges("tell_joke", route)
builder.add_edge("critique", "tell_joke")
builder.set_entry_point("tell_joke")
joke_graph = builder.compile()

# %%
from IPython.display import Image

Image(joke_graph.get_graph().draw_png())

# %%
for step in joke_graph.stream({"messages": [("user", "讲一个关于减肥的笑话")]}):
    print(step)

# %% [markdown]
# ## Main Graph
# 
# 主图只是一个路由器，它将消息发送到笑话图或直接响应。

# %%
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

class AssistantState(TypedDict):
    conversation: Annotated[List, operator.add]


# assistant_llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个得力的助手"),
        MessagesPlaceholder(variable_name="conversation"),
    ]
)


def add_to_conversation(message):
    return {"conversation": [message]}


main_builder = StateGraph(AssistantState)
main_builder.add_node(
    "assistant", assistant_prompt | llm | add_to_conversation
)


def get_user_message(state: AssistantState):
    last_message = state["conversation"][-1]
    # Convert to sub-graph state
    return {"messages": [last_message]}


def get_joke(state: SubGraphState):
    final_joke = state["messages"][-1]
    return {"conversation": [final_joke]}


main_builder.add_node("joke_graph", get_user_message | joke_graph | get_joke)


def route(state: AssistantState):
    message = state["conversation"][-1][-1]

    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """你是一个意图识别的助手，能够识别以下意图：
1. 讲故事
2. 讲笑话
3. AI绘画
4. 学习知识
5. 其他

例如：
用户输入：给我说个故事把。
1
用户输入：给我画个美女图片。
3

------
用户输入：{input}

请识别用户意图，返回上面意图的数字序号，只返回数字，不返回任何其他字符。
""")
        ]
    )

    chain = assistant_prompt | llm | output_parser

    result = chain.invoke({
        "input":message
    })

    result = result.strip()

    print("意图识别：",result)

    if result=="2" :
        return "joke_graph"
    else :
        return "assistant"


main_builder.set_conditional_entry_point(
    route,
)
main_builder.set_finish_point("assistant")
main_builder.set_finish_point("joke_graph")
graph = main_builder.compile()

# %%
Image(graph.get_graph().draw_png())

# %%
for step in graph.stream({"conversation": [("user", "请给我讲个减肥笑话")]}):
    print(step)

# %%
for step in graph.stream({"conversation": [("user", "推荐一点夏天的减肥餐")]}):
    print(step)

# %%



