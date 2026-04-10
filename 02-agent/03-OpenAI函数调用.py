# %% [markdown]
# ## Function calling 函数调用
# 
# 
# 与聊天完成 API 类似，助手 API 支持函数调用。函数调用允许您向助手描述函数，并让它智能地返回需要调用的函数及其参数。助手 API 在调用函数时将在运行期间暂停执行，你可以提供函数回调的结果以继续运行执行。
# 
# ### Defining functions 定义函数
# 
# 首先，在创建 Assistant 时定义您的函数：
# ```python
assistant = client.beta.assistants.create(
  instructions="You are a weather bot. Use the provided functions to answer questions.",
  model="gpt-4-turbo-preview",
  tools=[{
      "type": "function",
    "function": {
      "name": "getCurrentWeather",
      "description": "Get the weather in location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
          "unit": {"type": "string", "enum": ["c", "f"]}
        },
        "required": ["location"]
      }
    }
  }, {
    "type": "function",
    "function": {
      "name": "getNickname",
      "description": "Get the nickname of a city",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
        },
        "required": ["location"]
      }
    } 
  }]
)
# ```
# 
# ### 读取助手调用的函数
# 
# 当您启动触发函数的用户运行消息时，运行将进入状态 pending 。处理后，运行将进入一种 requires_action 状态，您可以通过检索运行来验证该状态。该模型可以使用并行函数调用提供多个函数一次调用：
# ```
{
  "id": "run_abc123",
  "object": "thread.run",
  "assistant_id": "asst_abc123",
  "thread_id": "thread_abc123",
  "status": "requires_action",
  "required_action": {
    "type": "submit_tool_outputs",
    "submit_tool_outputs": {
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "getCurrentWeather",
            "arguments": "{\"location\":\"San Francisco\"}"
          }
        },
        {
          "id": "call_abc456",
          "type": "function",
          "function": {
            "name": "getNickname",
            "arguments": "{\"location\":\"Los Angeles\"}"
          }
        }
      ]
    }
  },
# ```
# 
# ### 提交函数输出
# 
# 然后，可以通过提交调用函数的工具输出来完成运行。传递上述 required_action 对象中的 tool_call_id 引用，以匹配每个函数调用的输出。
# ```python
run = client.beta.threads.runs.submit_tool_outputs(
  thread_id=thread.id,
  run_id=run.id,
  tool_outputs=[
      {
        "tool_call_id": call_ids[0],
        "output": "22C",
      },
      {
        "tool_call_id": call_ids[1],
        "output": "LA",
      },
    ]
)
# ```
# 提交输出后，运行将进入 queued 状态，然后继续执行。
# 
# 

# %% [markdown]
# ```python
ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'], 
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
            prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')
        ), 
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['input'], template='{input}'
            )
        ), 
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)
# ```

# %% [markdown]
# ### agent_scratchpad
# 代理提示符必须有一个“agent_scratchpad”键，该键是“消息占位符”。 中间代理动作和工具输出消息将在这里传递。

# %% [markdown]
# ```python
RunnableAssign(
    mapper={
      agent_scratchpad: RunnableLambda(
          lambda x: format_to_openai_function_messages(x['intermediate_steps'])
      )
    }
)
| ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input'], 
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
            prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')
        ), 
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)
| RunnableBinding(
    bound=ChatOpenAI(
        client=<openai.resources.chat.completions.Completions object at 0x000001A5F7692890>, 
        async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x000001A59ACD2560>, 
        temperature=0.3, 
        openai_api_key=SecretStr('**********'), 
        openai_api_base='http://127.0.0.1:1234/v1', 
        openai_proxy=''
    ), 
    kwargs={
        'functions': [
            {
                'name': 'weather', 
                'description': '根据城市获取天气数据',
                'parameters': {
                    'properties': {
                        '__arg1': {
                            'title': '__arg1', 
                            'type': 'string'
                        }
                    }, 
                    'required': ['__arg1'], 
                    'type': 'object'
                }
            }
        ]
    }
)
| OpenAIFunctionsAgentOutputParser()
# ```

# %% [markdown]
# ## format_to_openai_function_messages
# 
# 将（AgentAction，工具输出）元组转换为 FunctionMessage。
# 
# - 参数：
# 
# middle_steps：LLM迄今为止采取的步骤以及观察结果
# 
# - 返回：
# 
# 发送给 LLM 进行下一次预测的消息列表
# 
# ```PYTHON
def format_to_openai_function_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    messages = []

    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages
# ```

# %% [markdown]
# ## _convert_agent_action_to_messages
# 
# 将代理操作转换为消息。此代码用于根据代理操作重建原始 AI 消息。
# 
# - 参数：
# 
# agent_action：要转换的代理操作。
# 
# - 返回：
# 
# 与原始工具调用相对应的 AIMessage。
# 
# ```PYTHON
if isinstance(agent_action, AgentActionMessageLog):
        return list(agent_action.message_log) + [
            _create_function_message(agent_action, observation)
        ]
    else:
        return [AIMessage(content=agent_action.log)]
# ```
# 
# 
# ## _create_function_message
# 
# 将代理操作和观察转换为功能消息。
# - 参数：
#     1. agent_action：来自代理的工具调用请求
#     2. 观察：工具调用的结果
# 
# 
# 
# - 返回：与原始工具调用相对应的 FunctionMessage
# 
# ```python
return FunctionMessage(
    name=agent_action.tool,
    content=content,
)
# ```

# %%



