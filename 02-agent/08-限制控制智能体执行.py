# %%
from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.1,
)
llm = model

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
    """构建思考过程。"""
    thoughts: List[BaseMessage] = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(
            content=template_tool_response.format(input=query,observation=observation)
        )
        thoughts.append(human_message)
    return thoughts

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
tools = [get_word_length,get_weather]

# %%
from langchain.tools.render import  render_text_description

prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
)
prompt

# %%
from langchain.agents.agent import AgentOutputParser
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentAction, AgentFinish
class JSONAgentOutputParser(AgentOutputParser):

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
agent_executor.invoke({"input": "广州的天气如何？"})

# %%
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,return_intermediate_steps=True,)

# %%
agent_executor.invoke({"input": "广州的天气如何？"})

# %%
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,max_iterations=2)
agent_executor.invoke({"input": "广州的天气如何？"})

# %%
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_execution_time=1,)
agent_executor.invoke({"input": "广州的天气如何？"})

# %%



