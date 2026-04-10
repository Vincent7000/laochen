# %% [markdown]
# # Branching
# 
# LangGraph 本身支持多分支操作。
# 
# - 项目管理：项目经理需要同时考虑项目的多个方面，如时间管理、资源分配、团队协作、风险管理等。
# 
# - 
# 决策制定：在商业或政策制定中，决策者需要考虑多个因素和潜在后果，如经济影响、社会效应、环境影响等
# 
# - 
# 多任务处理：在工作中，员工经常需要同时处理多个任务，如回复电子邮件、准备报告、参加会
# 
# - 。
# 研究和分析：研究人员在分析数据时，需要考虑不同的变量和它们之间的关系，同时保持对研究目标的关注。
# 
# ![image.png](attachment:86b5ccaa-ff72-4a13-9c9b-2d194d1d046b.png)

# %%
from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"
# llm = ChatOpenAI(
#     openai_api_key=openai_api_key,
#     openai_api_base=openai_api_base,
#     temperature=0.3,
#     model_kwargs = {
#         "frequency_penalty":0.9,
#         "presence_penalty":0.9
#     }
# )
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
    model="/mnt/d/ai/download/Smaug-34B-v0.1-AWQ"
)

# %%
# from langchain_anthropic import ChatAnthropic
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import MessageGraph

# %% [markdown]
# #### Define the logic

# %%
from langchain_core.messages import HumanMessage

# llm = ChatAnthropic(model="claude-3-haiku-20240307")


## Branch 1
fan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """现在你放弃一切想法，假装成为一个任何主题的狂热粉丝和追随者，应该尽一切能力吹捧主题的观点"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

proponent = fan_prompt | llm


## Branch 2
detractor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "现在你假装成为任何话题的批评者和坚定的反对者，应该尽一切能力提供有力的证据来批判和反驳主题的观点",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
opponent = detractor_prompt | llm


## Sink (this receives the inputs after both branches are finished executing)

synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "哪个论证更有力？ 选择一边。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def merge_messages(messages: list):
    print(list[-1])
    print(list[-2])
    original = messages[0].content
    arguments = "\n".join(
        [f"Argument {i}: {msg.content}" for i, msg in enumerate(messages[1:])]
    )
    return {
        "messages": [
            HumanMessage(
                content=f"""Topic: {original}
Arguments: {arguments}\n\n哪一个论点更有说服力？"""
            )
        ]
    }


final = merge_messages | synthesis_prompt | llm

# %%
proponent.invoke([HumanMessage(content="躺平是当代人的解药")])

# %%
opponent.invoke([HumanMessage(content="躺平是当代人的解药")])

# %% [markdown]
# ## Define Graph

# %%
builder = MessageGraph()


def dictify(messages: list):
    return {"messages": messages}


builder.add_node("source", lambda x: [])
builder.add_node("branch_1", dictify | proponent)
builder.add_node("branch_2", dictify | opponent)
builder.add_node("sink", final)

# Define edges

builder.set_entry_point("source")
# Fan out
builder.add_edge("source", "branch_1")
builder.add_edge("source", "branch_2")
# Fan back in
builder.add_edge("branch_1", "sink")
builder.add_edge("branch_2", "sink")

builder.set_finish_point("sink")
graph = builder.compile()

# %%
from IPython.display import Image

Image(graph.get_graph().draw_png())

# %%


# %%
graph.invoke([HumanMessage(content="躺平是当代人的解药")])

# %%
for step in graph.stream([HumanMessage(content="躺平是当代人的解药")]):
    print(step)

# %%
print(step["__end__"])

# %%



