# %% [markdown]
# # 计划和执行
# 
# 计划和执行代理通过首先计划要做什么，然后执行子任务来实现目标。 这个想法很大程度上受到[BabyAGI](https://github.com/yoheinakajima/babyagi)和[“Plan-and-Solve”论文](https://arxiv.org/abs/2305.04091)的启发。
# 
# 规划几乎总是由LLM完成。
# 
# 执行通常由单独的代理（配备工具）完成。

# %%
# 使用 utils 中的 searxng_search 替代 DuckDuckGo（国内网络无法访问 DuckDuckGo）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utils import searxng_search as search

# 测试搜索功能
# result = search.invoke("刘亦菲")
# print(result)

# %% [markdown]
# ## 加载LLM
# 
# 首先，让我们加载将用于控制代理的语言模型。

# %%
from langchain_openai import ChatOpenAI, OpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utils import llm, render_text_description

# %% [markdown]
# ## 定义工具
# 
# 接下来，让我们定义一些要使用的工具。让我们编写一个非常简单的 Python 函数来计算传入的单词的长度。
# 
# 请注意，这里我们使用的函数文档字符串非常重要。在此处阅读有关为什么会这样的更多信息

# %%
# from langchain.agents import tool
from langchain_core.tools import tool


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


get_word_length.invoke("abc")

# %%
import requests

@tool
def get_weather(location):
    """根据城市获取天气数据"""
    api_key = "SKcA5FGgmLvN7faJi"
    url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
    response = requests.get(url)
    #print(location)
    if response.status_code == 200:
        data = response.json()
        #print(data)
        weather = {
            'description':data['results'][0]["now"]["text"],
            'temperature':data['results'][0]["now"]["temperature"]
        }
        return weather
    else:
        raise Exception(f"失败接收天气信息：{response.status_code}")
            
print(f"广州天气: {get_weather.invoke('广州')}")

# %%
@tool
def search(keywords):
    """使用duckduckgo搜索引擎获取内容"""
    return search.run(keywords)

# %%
tools = [get_word_length,get_weather,search]

# %% [markdown]
# ## 创建提示
# 
# 现在让我们创建提示。由于 OpenAI 函数调用针对工具使用进行了微调，因此我们几乎不需要任何关于如何推理或如何输出格式的说明。我们将只有两个输入变量： input 和 agent_scratchpad 。 input 应为包含用户目标的字符串。 agent_scratchpad 应该是包含先前代理工具调用和相应工具输出的消息序列。

# %%
promptTemplate = """尽可能的帮助用户回答任何问题。

您可以使用以下工具来帮忙解决问题，如果已经知道了答案，也可以直接回答：

{tools}

回复格式说明
----------------------------

回复我时，请以以下两种格式之一输出回复：

选项 1：如果您希望人类使用工具，请使用此选项。
采用以下JSON模式格式化的回复内容：

```json
{{
    "reason": string, \\ 叙述使用工具的原因
    "action": string, \\ 要使用的工具。 必须是 {tool_names} 之一
    "action_input": string \\ 工具的输入
}}
````

选项2：如果您认为你已经有答案或者已经通过使用工具找到了答案，想直接对人类做出反应，请使用此选项。 采用以下JSON模式格式化的回复内容：

```json
{{
  "action": "Final Answer",
  "answer": string \\最终答复问题的答案放到这里！
}}
````

用户的输入
--------------------
这是用户的输入（请记住通过单个选项，以JSON模式格式化的回复内容，不要回复其他内容）：

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
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# %%
# 新版 LangChain 1.x 中 render_text_description 已移除，手动实现


prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
)

prompt

# %%
render_text_description(list(tools))

# get_word_length: get_word_length(word: str) -> int - Returns the length of a word.
# get_weather: get_weather(location) - 根据城市获取天气数据
# searxng_search: searxng_search(query) - 输入搜索内容，使用 SearXNG 进行搜索。

# %% [markdown]
# ## 将工具绑定到LLM
# 
# 在这种情况下，我们依赖于 OpenAI 工具调用 LLMs，它将工具作为一个单独的参数，并经过专门训练，知道何时调用这些工具。
# 
# 要将我们的工具传递给代理，我们只需要将它们格式化为 OpenAI 工具格式并将它们传递给我们的模型。（通过 bind 对函数进行 -ing，我们确保每次调用模型时都传入它们。

# %% [markdown]
# ## 创建代理
# 
# 将这些部分放在一起，我们现在可以创建代理。我们将导入最后两个实用程序函数：一个用于格式化中间步骤（代理操作、工具输出对）以输入可发送到模型的消息的组件，以及一个用于将输出消息转换为代理操作/代理完成的组件。

# %%
# 本地定义 TEMPLATE_TOOL_RESPONSE（新版 LangChain 已移除该导入）
TEMPLATE_TOOL_RESPONSE = """工具响应：
---------------------
{observation}

用户的输入：
---------------------
请根据工具的响应判断，是否能够回答问题：

{input}

请根据工具响应的内容，思考接下来回复。回复格式严格按照前面所说的2种JSON回复格式，选择其中1种进行回复。请记住通过单个选项，以JSON模式格式化的回复内容，不要回复其他内容。"""

# %%
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
def format_log_to_messages(
    query,
    intermediate_steps,
    template_tool_response,
):
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(input=query,observation=observation)
        )
        thoughts.append(human_message)
    return thoughts

# %%
from langchain_core.output_parsers import BaseOutputParser as AgentOutputParser
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
class JSONAgentOutputParser(AgentOutputParser):
    """以 JSON 格式解析工具调用和最终答案。

     期望输出采用两种格式之一。

     如果输出信号表明应采取行动，
     应采用以下格式。 这将导致 AgentAction
     被退回。

    ```
    {
      "action": "search",
      "action_input": "2+2"
    }
    ```
    如果输出信号表明应给出最终答案，
    应采用以下格式。 这将导致 AgentFinish
    被退回。

    ```
    {
      "action": "Final Answer",
      "answer": "4"
    }
    ```
    """

    def parse(self, text):
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                # gpt turbo frequently ignores the directive to emit a single action
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
# from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough


agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["input"],
                x["intermediate_steps"], 
                template_tool_response=TEMPLATE_TOOL_RESPONSE
            )
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

# %%
# 新版 LangChain 1.x 中 AgentExecutor 已移除，使用 langchain.agents 的 create_agent
try:
    from langchain.agents import create_agent
    agent_executor = create_agent(
        model=llm,
        tools=tools,
    )
    # 注释掉旧版调用方式
    # agent_executor.invoke({"input": "英国现任首相是谁？"})
except ImportError:
    print("create_agent 不可用，跳过 AgentExecutor 测试")

# %% [markdown]
# ## 任务计划与监督者

# %%
planPrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """让我们首先了解问题并制定解决问题的计划。 
请输出以标题“计划：”开头的计划，然后是编号的步骤列表。 
请制定准确完成任务所需的最少步骤数的计划，计划里不要写上任何答案，也不要将答案放到括号里。

如果任务是一个问题，那么最后一步总是“鉴于采取了上述步骤，请回答用户最初的问题”。
""",
        ),
        ("user", "{input}"),
        #MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# %%
import re
def planParse(message):
    text = message.content
    steps = [v for v in re.split("\n\s*\d+\. ", text)[1:]]
    return steps

# %%
planChain = planPrompt | llm | planParse

# %%
print(f"{planChain.invoke({'input':'写一篇关于历代美国总统的年龄跟发布政策偏好关系'})}")

# %%
splitTaskPrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """让我们首先了解用户问题或者是计划，并根据所提供的工具判断是否能直接通过工具解决问题或者计划。
            
如果需要拆分问题或者是计划，请输出以标题“计划：”开头的计划，然后是编号的步骤列表。 

否则不需要拆分，请直接输出<End>，不用返回任何内容。

工具：
{tools}



""",
        ),
        ("user", "{input}"),
        #MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

splitTaskPrompt = splitTaskPrompt.partial(
        tools=render_text_description(list(tools)),
)
splitTaskPrompt

# %%
splitTaskChain = splitTaskPrompt | llm | planParse
print(f"{splitTaskChain.invoke({'input':'收集历任美国总统的个人信息和政治生涯信息（出生日期、就职日期、主要政策等）。'})}")
# print(splitTaskChain.invoke({"input":"收集历任美国总统的个人信息和政治生涯信息（出生日期、就职日期、主要政策等）。"}))

# %%
print(f"{splitTaskChain.invoke({'input':'搜索每个总统的名字，并查看他们的维基百科页面。'})}")
# splitTaskChain.invoke({"input":"搜索每个总统的名字，并查看他们的维基百科页面。"})

# %%



