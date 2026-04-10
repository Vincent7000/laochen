# %% [markdown]
# ## JSON 聊天代理
# 
# 一些语言模型特别擅长编写 JSON。此代理使用 JSON 来格式化其输出，旨在支持聊天模型。
from langchain.agents import  AgentExecutor, create_json_chat_agent
promt = hub.pull("hwchase17/react-chat-json")
print(promt)
# %% [markdown]
# ```python
ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'], 
    input_types={
        'chat_history': typing.List[
            typing.Union[
                langchain_core.messages.ai.AIMessage, 
                langchain_core.messages.human.HumanMessage, 
                langchain_core.messages.chat.ChatMessage, 
                langchain_core.messages.system.SystemMessage, 
                langchain_core.messages.function.FunctionMessage, 
                langchain_core.messages.tool.ToolMessage
            ]
        ], 
        'agent_scratchpad': typing.List[
            typing.Union[
                langchain_core.messages.ai.AIMessage,
                langchain_core.messages.human.HumanMessage, 
                langchain_core.messages.chat.ChatMessage, 
                langchain_core.messages.system.SystemMessage, 
                langchain_core.messages.function.FunctionMessage, 
                langchain_core.messages.tool.ToolMessage
            ]
        ]
    }, 
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[], 
                template='Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'
            )
        ), 
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input', 'tool_names', 'tools'], 
                template='TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:\n\n{tools}\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": string, \\ The action to take. Must be one of {tool_names}\n    "action_input": string \\ The input to the action\n}}\n```\n\n**Option #2:**\nUse this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": "Final Answer",\n    "action_input": string \\ You should put what you want to return to use here\n}}\n```\n\nUSER\'S INPUT\n--------------------\nHere is the user\'s input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n{input}'
            )
        ), 
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)
# ```

# %% [markdown]
# ## system prompt
# 
# Assistant is a large language model trained by OpenAI.
# 
# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
# 
# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
# 
# Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
# 
# 
# Assistant是OpenAI训练的大型语言模型。
# 
# Assistant 旨在帮助完成各种任务，从回答简单的问题到就各种主题提供深入的解释和讨论。 作为一种语言模型，Assistant 能够根据接收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供与当前主题连贯且相关的响应。
# 
# Assistant 不断学习和改进，其能力也在不断发展。 它能够处理和理解大量文本，并可以利用这些知识对各种问题提供准确且内容丰富的答案。 此外，Assistant 能够根据收到的输入生成自己的文本，使其能够参与讨论并就各种主题提供解释和描述。
# 
# 总的来说，Assistant 是一个功能强大的系统，可以帮助完成广泛的任务，并提供有关广泛主题的宝贵见解和信息。 无论您需要解决特定问题的帮助还是只想就特定主题进行对话，助理都会随时为您提供帮助。

# %% [markdown]
# ## Human prompt
# 
# 
# TOOLS
# 
# ------
# Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
# 
# {tools}
# 
# RESPONSE FORMAT INSTRUCTIONS
# 
# ----------------------------
# 
# When responding to me, please output a response in one of two formats:
# 
# **Option 1:**\nUse this if you want the human to use a tool.
# Markdown code snippet formatted in the following schema:
# 
# ```json
# {{
#     "action": string, \\ The action to take. Must be one of {tool_names}
#     "action_input": string \\ The input to the action
#  }}
#  ```
#  
#  **Option #2:**
#  Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:
#  
#  ```json
#  {{
#      "action": "Final Answer",
#      "action_input": string \\ You should put what you want to return to use here
#  }}
#  ```
#  
#  USER\'S INPUT
#  
#  --------------------
#  
#  Here is the user\'s input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):
#  
#  {input}

# %% [markdown]
# ## 人工提示
# 
# 
# 工具
# 
# ------
# 助理可以要求用户使用工具查找可能有助于回答用户原始问题的信息。 人类可以使用的工具有：
# 
# {tools}
# 
# 回复格式说明
# 
# ----------------------------
# 
# 回复我时，请以以下两种格式之一输出回复：
# 
# **选项 1：**\n如果您希望人类使用工具，请使用此选项。
# 采用以下模式格式化的 Markdown 代码片段：
# 
# ```json
# {{
#      "action": string, \\ 要采取的操作。 必须是 {tool_names} 之一
#      "action_input": string \\ 操作的输入
#   }}
#   ````
#  
#   **选项#2：**
#   如果您想直接对人类做出反应，请使用此选项。 采用以下模式格式化的 Markdown 代码片段：
#  
#   ```json
#   {{
#       "action": "Final Answer",
#       "action_input": string \\ 你应该把你想要返回使用的东西放在这里
#   }}
#   ````
#  
#   用户的输入
#  
#   --------------------
#  
#   这是用户的输入（请记住通过单个操作使用 json blob 的 markdown 代码片段进行响应，仅此而已）：
#  
#   {input}

# %%
[
    {
      "role": "system",
      "content": "Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."
    },
    {
      "role": "user",
      "content": "TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:\n\nweather: 根据城市获取天气数据\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{\n    \"action\": string, \\ The action to take. Must be one of weather\n    \"action_input\": string \\ The input to the action\n}\n```\n\n**Option #2:**\nUse this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:\n\n```json\n{\n    \"action\": \"Final Answer\",\n    \"action_input\": string \\ You should put what you want to return to use here\n}\n```\n\nUSER'S INPUT\n--------------------\nHere is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n今天广州的天气"
    },
    {
      "role": "assistant",
      "content": "```json\n{\n    \"action\": \"weather\",\n    \"action_input\": \"广州\"\n}\n```"
    },
    {
      "role": "user",
      "content": "TOOL RESPONSE: \n---------------------\n{'description': '阴', 'temperature': '20'}\n\nUSER'S INPUT\n--------------------\n\nOkay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!"
    }
]

# %% [markdown]
# ## 中间的响应
# 
# TOOL RESPONSE: 
# 
# ---------------------
# 
# {'description': '阴', 'temperature': '20'}
# 
# USER'S INPUT
# 
# --------------------
# 
# Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!

# %% [markdown]
# ## 中间的响应
# 
# 工具响应：
# 
# --------------------
# 
# {'描述': '阴', '温度': '20'}
# 
# 用户的输入
# 
# --------------------
# 
# 好的，那么我上次评论的回应是什么？ 如果使用从工具中获得的信息，您必须明确提及，而不提及工具名称 - 我已经忘记了所有工具响应！ 请记住通过一个操作来响应 json blob 的 markdown 代码片段，而不是其他任何东西 - 即使您只想响应用户。 无论如何，请勿使用除 JSON 片段之外的任何内容进行响应！

# %%



