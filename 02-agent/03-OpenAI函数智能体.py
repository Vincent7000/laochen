
# ## OpenAI函数智能体
# 
# 某些 OpenAI 模型（如 gpt-3.5-turbo-0613 和 gpt-4-0613）已经过微调，可以检测何时应该调用函数，并使用应传递给函数的输入进行响应。在 API 调用中，您可以描述函数，并让模型智能地选择输出包含用于调用这些函数的参数的 JSON 对象。OpenAI 函数 API 的目标是比通用文本补全或聊天 API 更可靠地返回有效和有用的函数调用。
# 
# 许多开源模型都采用了相同的函数调用格式，并且还对模型进行了微调，以检测何时应该调用函数。
# 
# OpenAI Functions Agent 旨在与这些模型配合使用。
# 
# 安装 openai ， tavily-python 因为 LangChain 软件包在内部调用它们。


from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)
llm = model


from langchain.tools import BaseTool
import requests
from typing import Any

class Weather(BaseTool):
    # name = "weather"
    # description = "根据城市获取天气数据"
    name: str = "weather"
    description: str = "根据城市获取天气数据"
    
    def __init__(self):
        super().__init__()
        
#     async def _arun(self,*args,**kwargs):
#         pass
        
    def get_weather(self,location):
        api_key = "SKcA5FGgmLvN7faJi"
        url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
        response = requests.get(url)
        print(location)
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
    def _run(self,para):
        print(para)
        return self.get_weather(para)
            
weather_tool = Weather()
weather_tool.run({"para":"广州"})


from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt


tools = [weather_tool]
# Construct the OpenAI Functions agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor


agent_executor.invoke({"input": "今天广州的天气"})


# 
# ```json
# {
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful assistant"
#     },
#     {
#       "role": "user",
#       "content": "今天广州的天气"
#     }
#   ],
#   "model": "gpt-3.5-turbo",
#   "functions": [
#     {
#       "name": "weather",
#       "description": "根据城市获取天气数据",
#       "parameters": {
#         "properties": {
#           "__arg1": {
#             "title": "__arg1",
#             "type": "string"
#           }
#         },
#         "required": [
#           "__arg1"
#         ],
#         "type": "object"
#       }
#     }
#   ],
#   "n": 1,
#   "stream": true,
#   "temperature": 0.3
# }
# ```


# ## OpenAI 工具
# 
# 较新的 OpenAI 模型已经过微调，可以检测何时应该调用一个或多个函数，并使用应传递给函数的输入进行响应。在 API 调用中，您可以描述函数，并让模型智能地选择输出包含参数的 JSON 对象来调用这些函数。OpenAI 工具 API 的目标是比使用通用文本补全或聊天 API 更可靠地返回有效和有用的函数调用。
# 
# OpenAI 将调用单个函数的能力称为函数，将调用一个或多个函数的能力称为工具。
# 
# 在 OpenAI 聊天 API 中，函数现在被视为已弃用的旧选项，取而代之的是工具。
# 
# 如果您使用 OpenAI 模型创建代理，则应使用此 OpenAI 工具代理，而不是 OpenAI 函数代理。
# 
# 使用工具允许模型在适当的时候请求调用多个函数。
# 
# 在某些情况下，这有助于显著减少代理实现其目标所需的时间。


from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt


from langchain.agents import AgentExecutor, create_openai_tools_agent
# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "今天广州的天气"})


# ```json
# {
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful assistant"
#     },
#     {
#       "role": "user",
#       "content": "今天广州的天气"
#     }
#   ],
#   "model": "gpt-3.5-turbo",
#   "n": 1,
#   "stream": true,
#   "temperature": 0.3,
#   "tools": [
#     {
#       "type": "function",
#       "function": {
#         "name": "weather",
#         "description": "根据城市获取天气数据",
#         "parameters": {
#           "properties": {
#             "__arg1": {
#               "title": "__arg1",
#               "type": "string"
#             }
#           },
#           "required": [
#             "__arg1"
#           ],
#           "type": "object"
#         }
#       }
#     }
#   ]
# }
# ```


# ## XML 代理
# 
# 一些语言模型（如Anthropic的Claude）特别擅长推理/编写XML。这将介绍如何在提示时使用使用 XML 的代理。
# 
# 仅与非结构化工具一起使用;即接受单个字符串输入的工具。
# 
# 


tools = [weather_tool]


from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/xml-agent-convo")
prompt


from langchain.agents import AgentExecutor, create_xml_agent

# Choose the LLM that will drive the agent
llm = model

# Construct the XML agent
agent = create_xml_agent(llm, tools, prompt)
agent


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "今天广州的天气"})


# You are a helpful assistant. Help the user answer any questions.\n\nYou have access to the following tools:\n\nweather: 根据城市获取天气数据\n\nIn order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>\nFor example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:\n\n<tool>search</tool><tool_input>weather in SF</tool_input>\n<observation>64 degrees</observation>\n\nWhen you are done, respond with a final answer between <final_answer></final_answer>. For example:\n\n<final_answer>The weather in SF is 64 degrees</final_answer>\n\nBegin!\n\nPrevious Conversation:\n\n\nQuestion: 今天广州的天气\n


# You are a helpful assistant. Help the user answer any questions.\n\nYou have access to the following tools:\n\nweather: 根据城市获取天气数据\n\nIn order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>\nFor example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:\n\n<tool>search</tool><tool_input>weather in SF</tool_input>\n<observation>64 degrees</observation>\n\nWhen you are done, respond with a final answer between <final_answer></final_answer>. For example:\n\n<final_answer>The weather in SF is 64 degrees</final_answer>\n\nBegin!\n\nPrevious Conversation:\n\n\nQuestion: 今天广州的天气\n<tool>weather</tool><tool_input>广州</tool_input><observation>{'description': '阴', 'temperature': '21'}</observation>


# ## JSON 聊天代理
# 
# 一些语言模型特别擅长编写 JSON。此代理使用 JSON 来格式化其输出，旨在支持聊天模型。


from langchain.agents import AgentExecutor, create_json_chat_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react-chat-json")
prompt


# Construct the JSON agent
agent = create_json_chat_agent(llm, tools, prompt)
agent


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


agent_executor.invoke({"input": "今天广州的天气"})


# ```json
# {
#   "messages": [
#     {
#       "role": "system",
#       "content": "Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."
#     },
#     {
#       "role": "user",
#       "content": "TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:\n\nweather: 根据城市获取天气数据\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{\n    \"action\": string, \\ The action to take. Must be one of weather\n    \"action_input\": string \\ The input to the action\n}\n```\n\n**Option #2:**\nUse this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:\n\n```json\n{\n    \"action\": \"Final Answer\",\n    \"action_input\": string \\ You should put what you want to return to use here\n}\n```\n\nUSER'S INPUT\n--------------------\nHere is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n今天广州的天气"
#     }
#   ],
#   "model": "gpt-3.5-turbo",
#   "n": 1,
#   "stop": [
#     "\nObservation"
#   ],
#   "stream": true,
#   "temperature": 0.3
# }
# ```


# ```python
# 
# {
#   "messages": [
#     {
#       "role": "system",
#       "content": "Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."
#     },
#     {
#       "role": "user",
#       "content": "TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:\n\nweather: 根据城市获取天气数据\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{\n    \"action\": string, \\ The action to take. Must be one of weather\n    \"action_input\": string \\ The input to the action\n}\n```\n\n**Option #2:**\nUse this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:\n\n```json\n{\n    \"action\": \"Final Answer\",\n    \"action_input\": string \\ You should put what you want to return to use here\n}\n```\n\nUSER'S INPUT\n--------------------\nHere is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n今天广州的天气"
#     },
#     {
#       "role": "assistant",
#       "content": "```json\n{\n    \"action\": \"weather\",\n    \"action_input\": \"广州\"\n}\n```"
#     },
#     {
#       "role": "user",
#       "content": "TOOL RESPONSE: \n---------------------\n{'description': '阴', 'temperature': '20'}\n\nUSER'S INPUT\n--------------------\n\nOkay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!"
#     }
#   ],
#   "model": "gpt-3.5-turbo",
#   "n": 1,
#   "stop": [
#     "\nObservation"
#   ],
#   "stream": true,
#   "temperature": 0.3
# }
# 
# ```


# ## ReAct Agent
# 
# 
# ReAct(reason, act)整合了推理和行动能力，是一种使用自然语言推理解决复杂任务的语言模型。ReAct旨在用于允许LLM执行某些操作的任务。例如，在MRKL系统中，LLM可以与外部API交互以检索信息。当提出问题时，LLM可以选择执行操作以检索信息，然后根据检索到的信息回答问题。
# 
# ReAct系统可以被视为具有推理和行动能力的MRKL系统。
# 给机器人小明一个任务：去厨房为你做一杯咖啡。小明不仅要完成这个任务，还要告诉你他是如何一步步完成的。
# 
# ### 缺乏ReAct的机器人系统：
# 
# 1. 小明直接跑到厨房。
# 2. 你听到了一些声音，但不知道小明在做什么。
# 3. 过了一会儿，小明回来给你一杯咖啡。
# 4. 这样的问题是，你不知道小明是怎么做咖啡的，他是否加了糖或奶，或者他是否在过程中遇到了任何问题。
# 
# ### ReAct的机器人系统：
# 
#     - Thought 1：“我现在去厨房。”
#     - Act 1：“启动进入厨房。”
#     - Obs 1：“看到厨房内的环境。”
# 
# 
#     - Thought 2：“我需要找到咖啡粉和咖啡机。”
#     - Act 2：“寻找咖啡粉和咖啡机。”
#     - Obs 2：“找到了咖啡粉和咖啡机。”
# 
# 
#     - Thought 3：“我现在可以开始煮咖啡了。”
#     - Act 3：“开始煮咖啡。”
#     - Obs 3：“咖啡做好了。”
# 
# 
#     - Thought 4：“可以把咖啡送过去了。”
#     - Act 4：“送咖啡。”
#     - Obs 4：“得到了表扬。”
# 
# ReAct能够首先通过推理问题（Thought 1），然后执行一个动作（Act 1）。然后它收到了一个观察（Obs 1），并继续进行这个思想，行动，观察循环，直到达到结论。


from langchain.agents import AgentExecutor, create_react_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")
prompt


# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor


agent_executor.invoke({"input": "今天广州的天气"})


# ```
# {
#   "messages": [
#     {
#       "role": "user",
#       "content": "Answer the following questions as best you can. You have access to the following tools:\n\nweather: 根据城市获取天气数据\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [weather]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: 今天广州的天气\nThought:Thought: I need to find out the weather in Guangzhou today.\nAction: [weather]\nAction Input: \"Guangzhou\", \"today\"\nObservation: [weather] is not a valid tool, try one of [weather].\nThought: I apologize for the confusion earlier. Here's the correct response to your question:\n\nQuestion: 今天广州的天气\nThought: I need to find out the weather in Guangzhou today.\nAction: [weather]\nAction Input: \"Guangzhou\", \"today\"\nObservation: [weather] is not a valid tool, try one of [weather].\nThought: I apologize for the confusion earlier. Here's the correct response to your question:\n\nQuestion: 今天广州的天气\nThought: I need to find out the weather in Guangzhou today.\nAction: [weather]\nAction Input: \"Guangzhou\", \"today\"\nObservation: [weather] is not a valid tool, try one of [weather].\nThought: "
#     }
#   ],
#   "model": "gpt-3.5-turbo",
#   "n": 1,
#   "stop": [
#     "\nObservation"
#   ],
#   "stream": true,
#   "temperature": 0.3
# }
# ```


from langchain.agents import AgentExecutor, create_self_ask_with_search_agent


# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/self-ask-with-search")
prompt


# Construct the Self Ask With Search Agent
agent = create_self_ask_with_search_agent(llm, tools, prompt)
agent


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "今天广州的天气"})





