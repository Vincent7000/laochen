# %% [markdown]
# ## shell
# 
# Shell命令是操作系统中用于在命令行界面（CLI）中执行各种操作的工具。它们可以用于文件和目录管理（如创建、删除、移动和复制）、查看和编辑文本文件、管理系统权限和所有权、监控系统资源、进行网络通信和远程管理，以及安装和更新软件包等。掌握Shell命令可以极大地提高工作效率和系统的管理能力。
# 

# %%
from langchain.tools import ShellTool

shell_tool = ShellTool()

shell_tool.run({"commands": ["echo 'hello world'"]})

# %%
shell_tool.run({"commands": ["start chrome"]})

# %%
shell_tool.run({"commands": ["start chrome.exe http://www.baidu.com/s?wd=仙剑四"]})

# %%
shell_tool.run({"commands": ["mkdir newtxt"]})

# %%
from langchain.agents import tool


@tool
def run_shell(command: str) -> int:
    """在电脑上运行shell的命令操控电脑，例如输入command命令打开浏览器用百度搜索郭德纲，则command命令为 start chrome http://www.baidu.com/s?wd=郭德纲"""
    return shell_tool.run({"commands": [command]})


run_shell.invoke("mkdir 老陈文件夹")

# %% [markdown]
# ## 加载LLM
# 
# 首先，让我们加载将用于控制代理的语言模型。

# %%
from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)
llm = model

# %% [markdown]
# ## 定义工具
# 
# 接下来，让我们定义一些要使用的工具。让我们编写一个非常简单的 Python 函数来计算传入的单词的长度。
# 
# 请注意，这里我们使用的函数文档字符串非常重要。在此处阅读有关为什么会这样的更多信息

# %%
from langchain.agents import tool


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
            
get_weather.invoke("广州")

# %%
tools = [get_word_length,get_weather,run_shell]

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
from langchain.tools.render import  render_text_description

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
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE

TEMPLATE_TOOL_RESPONSE

# %%
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
from langchain.agents.agent import AgentOutputParser
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
class JSONAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in JSON format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    {
      "action": "search",
      "action_input": "2+2"
    }
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

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
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%
agent_executor.invoke({"input": "当前目录创建1个老陈文件夹"})

# %%
agent_executor.invoke({"input": "打开浏览器使用百度搜索黑悟空"})

# %% [markdown]
# ## 添加历史记录
# 
# 这太棒了 - 我们有一个代理！但是，此代理是无状态的 - 它不记得有关先前交互的任何信息。这意味着你不能轻易地提出后续问题。让我们通过添加内存来解决这个问题。
# 
# 为此，我们需要做两件事：
# 
# 1. 在提示符中为内存变量添加一个位置
# 2. 跟踪聊天记录
# 
# 首先，让我们在提示中添加一个内存位置。为此，我们通过为带有 键 "chat_history" 的消息添加一个占位符 .请注意，我们将其放在新用户输入的上方（以遵循对话流）。

# %%
from langchain.prompts import MessagesPlaceholder


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是非常强大的助手，你可以使用各种工具来完成人类交给的问题和任何。",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", promptTemplate),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# %%
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

# %%
prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
)

agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(
                x["input"],
                x["intermediate_steps"], 
                template_tool_response=TEMPLATE_TOOL_RESPONSE,
            )
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%


chat_history.extend(
    [
        HumanMessage(content="你好，我是老陈"),
        AIMessage(content="你好，老陈。很高兴认识你！"),
    ]
)
agent_executor.invoke({"input": "我叫什么名字？", "chat_history": chat_history})

# %%
chat_history.extend(
    [
        HumanMessage(content="刘德华的老婆是谁？"),
        AIMessage(content="刘德华的老婆是朱丽倩（Carol Chu）。"),
    ]
)
agent_executor.invoke({"input": "刘德华老婆有演过电影吗", "chat_history": chat_history})

# %%



