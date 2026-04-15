# %% [markdown]
# ![image.png](attachment:9a3df3a2-1e32-464b-81c2-3255f5d1a8e1.png)
# 
# 
# LangGraph 是一个有用于构建有状态和多角色的 Agents 应用，它并不是一个独立于 Langchain 的新框架，而是基于 Langchain 之上构建的一个扩展库，可以与 Langchain 现有的链（Chains）、LangChain Expression Language（LCEL）等无缝协作。LangGraph 能够协调多个 Chain、Agent、Tool 等共同协作来完成输入任务，支持 LLM 调用“循环”和 Agent 过程的更精细化控制。
# 
# LangGraph 的实现方式是把之前基于 AgentExecutor 的黑盒调用过程，用一种新的形式来构建：状态图（StateGraph）。把基于 LLM 的任务（比如：RAG、代码生成等）细节用 Graph 进行精确的定义（定义图的节点与边），最后基于这个图来编译生成应用。在任务运行过程中，维持一个中央状态对象(state)，会根据节点的跳转不断更新，状态包含的属性可自行定义
# 
# LangGraph发布，多代理工作流框架助力应用开发者在应用层搭建自己的多专家模型。LangGraph提供Python和JS版本，支持包含循环的LLM工作流创建。多代理工作流通过独立代理节点和连接表示，有助于开发概念模型。GPT-Newspaper和CrewAI两个基于LangGraph构建的第三方应用程序示例。

# %% [markdown]
# ## LangGraph 的几个基本概念：
# 
# StateGraph：代表整个状态图的基础类。
# 
# Nodes：节点。有了图之后，可以向图中添加节点，节点通常是一个可调用的函数、一个可运行的 Chain 或 Agent。有一个特殊的节点叫END，进入这个节点，代表运行结束。
# 
# 在上图中，推理函数调用、调用检索器、生成响应内容、问题重写等都是其中的任务节点。
# 
# Edges：边。有了节点后，需要向图中添加边，边代表从上一个节点跳转到下一个节点的关系。目前有三种类型的边：
# 
# Starting Edge、一种特殊的边。用来定义任务运行的开始节点，它没有上一个节点。
# 
# Normal Edge：普通边。代表上一个节点运行完成后立即进入下一个节点。比如在调用 Tools 后获得结果后，立刻进入 LLM 推理节点。
# 
# Conditional Edge：条件边。代表上一个节点运行完成后，需要根据条件跳转到某个节点，因此这种边不仅需要上游节点、下游节点，还需要一个条件函数，根据条件函数的返回来决定下游节点。
# 
# 在上图中，Check Relevance 就是一个条件边，它的上游节点是检索相关文档，条件函数是判断文档是否相关，如果相关，则进入下游节点【产生回答】；如果不相关，则进入下游节点【重写输入问题】。

# %%
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utils import llm, searxng_search

# 测试搜索工具
searxng_search.invoke("郭德纲")

# %%
promptTemplate = """尽可能的帮助用户回答任何问题。

您可以使用以下工具来帮忙解决问题，如果已经知道了答案，也可以直接回答：

searxng_search : searxng_search(query) -> 输入搜索内容，使用 LLM 返回对该主题的简单介绍。

回复格式说明
----------------------------

回复我时，请以以下两种格式之一输出回复：

选项 1：如果您希望人类使用工具，请使用此选项。
采用以下JSON模式格式化的回复内容,回复的格式里不要有注释内容：

```json
{{
    "reason": string, \\ 叙述使用工具的原因
    "action": "searxng_search", \\ 要使用的工具。 必须是 searxng_search
    "action_input": string \\ 工具的输入
}}
````

选项2：如果您认为你已经有答案或者已经通过使用工具找到了答案，想直接对人类做出反应，请使用此选项。 采用以下JSON模式格式化的回复内容,回复的格式里不要有注释内容：

```json
{{
  "action": "Final Answer",
  "answer": string \\最终答复问题的答案放到这里！
}}
````

用户的输入
--------------------
这是用户的输入（请记住通过单个选项，以JSON模式格式化的回复内容,回复的格式里不要有注释内容，不要回复其他内容）：

{input}

"""

# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。",
        ),
        ("user", promptTemplate),
    ]
)

# %%
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import StrOutputParser
class JSONAgentOutputParser(BaseOutputParser):

    def parse(self, text):
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                # 经常忽略发出单个动作的指令
                logger.warning("Got multiple action responses: %s", response)
                response = response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["answer"]}, text)
            else:
                return AgentAction(
                    response["action"], response.get("action_input", {}), text
                )
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "json-agent"

# %%
output_parser = StrOutputParser()
chain1 = prompt | llm 

chain1.invoke({"input":"小米su7的发布时间"})

# %%
chain1.invoke({"input":"请使用中文给我讲个笑话吧？"})

# %%
promptTemplate = """使用浏览器获取的搜索内容：
---------------------
{observation}
---------------------
请根据浏览器的响应，回答下面的问题：

{input}"""

prompt2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。",
        ),
        ("user", promptTemplate),
    ]
)


# %%
chain2 =   prompt2 | llm 

# %%
result = searxng_search.invoke("小米su7 发布时间")
result

# %%
chain2.invoke({"input":"小米su7的发布时间","observation":result})

# %%
from langgraph.graph import END, MessageGraph

graph = MessageGraph()

graph.add_node("chain", chain1)
graph.add_edge("chain", END)

graph.set_entry_point("chain")

runnable1 = graph.compile()

# %%
# 保存图为 PNG 文件
with open("graph1.png", "wb") as f:
    f.write(runnable1.get_graph().draw_png())
print("图已保存为 graph1.png")

# %%
# promptValue = prompt.invoke({"input":"小米su7的发布时间"})
from langchain_core.messages import HumanMessage
runnable1.invoke(HumanMessage("小米su7的发布时间"))

# %%
def process(state):
    print(state)
    content = state[-1].content
    return {"input":state[0].content,"observation":content}

chain2 =  process | prompt2 | llm 

# %%
from langgraph.graph import END, MessageGraph

graph = MessageGraph()

def tool(state):
    print(state)
    content = state[-1].content
    response = parse_json_markdown(content)

    result = searxng_search.invoke(response["action_input"])
    return HumanMessage(result)

graph.add_node("chain1", chain1)
graph.add_node("tool",tool)
graph.add_node("chain2", chain2)




# 设置开始
graph.set_entry_point("chain1")

# 设置条件边
def router(state):
    print(state)
    content = state[-1].content
    response = parse_json_markdown(content)
    if response["action"] == "Final Answer":
        return "end"
    else:
        return "tool"
graph.add_conditional_edges("chain1", router, {
    "tool": "tool",
    "end": END,
})

graph.add_edge("tool", "chain2")
graph.add_edge("chain2", END)




runnable2 = graph.compile()

# %%


# %%
# 保存图为 PNG 文件
with open("graph2.png", "wb") as f:
    f.write(runnable2.get_graph().draw_png())
print("图已保存为 graph2.png")

# %%
runnable2.invoke(HumanMessage("小米su7的发布时间"))

# %%


