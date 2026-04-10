# %% [markdown]
# ## XML 代理
# 
# 一些语言模型（如Anthropic的Claude）特别擅长推理/编写XML。这将介绍如何在提示时使用使用 XML 的代理。
# 
# 仅与非结构化工具一起使用;即接受单个字符串输入的工具。
# 
# 

# %% [markdown]
# ```python
ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tools'], 
    partial_variables={'chat_history': ''}, 
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[
                    'agent_scratchpad', 
                    'chat_history', 
                    'input', 
                    'tools'
                ], 
                template="You are a helpful assistant. Help the user answer any questions.\n\nYou have access to the following tools:\n\n{tools}\n\nIn order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>\nFor example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:\n\n<tool>search</tool><tool_input>weather in SF</tool_input>\n<observation>64 degrees</observation>\n\nWhen you are done, respond with a final answer between <final_answer></final_answer>. For example:\n\n<final_answer>The weather in SF is 64 degrees</final_answer>\n\nBegin!\n\nPrevious Conversation:\n{chat_history}\n\nQuestion: {input}\n{agent_scratchpad}"
            )
        )
    ]
)
# ```

# %%
prompt = '''You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:

{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

Previous Conversation:

{chat_history}

Question: {input}

{agent_scratchpad}

'''

# %% [markdown]
# 你是一个有用的助手。 帮助用户回答任何问题。
# 
# 您可以使用以下工具：
# 
# {tools}
# 
# 为了使用工具，您可以使用 `<tool></tool>`和 `<tool_input></tool_input>` 标签。 然后您将收到 `<observation></observation>` 形式的响应
# 例如，如果您有一个名为“search”的工具，可以运行谷歌搜索，为了搜索 SF 的天气，您将响应：
# ```
# <tool>search</tool><tool_input>旧金山天气</tool_input>
# <observation>64度</observation>
# ```
# 完成后，请在 `<final_answer></final_answer>` 之间回复最终答案。 例如：
# 
# `<final_answer>`旧金山的天气温度为 64 度`</final_answer>`
# 
# 开始！
# 
# 之前的对话：
# 
# {chat_history}
# 
# 问题：{input}
# 
# {agent_scratchpad}

# %%
prompt = '''You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:

weather: 根据城市获取天气数据

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

Previous Conversation:


Question: 今天广州的天气
'''

# %%
prompt = '''You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:

weather: 根据城市获取天气数据

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>
For example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:

<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>. For example:

<final_answer>The weather in SF is 64 degrees</final_answer>

Begin!

Previous Conversation:


Question: 今天广州的天气
<tool>weather</tool><tool_input>广州</tool_input><observation>{'description': '阴', 'temperature': '21'}</observation>

'''

# %% [markdown]
# 你是一个有用的助手。 帮助用户回答任何问题。
# 
# 您可以使用以下工具：
# 
# weather: 根据城市获取天气数据
# 
# 为了使用工具，您可以使用 `<tool></tool>`和 `<tool_input></tool_input>` 标签。 然后您将收到 `<observation></observation>` 形式的响应
# 例如，如果您有一个名为“search”的工具，可以运行谷歌搜索，为了搜索 SF 的天气，您将响应：
# ```
# <tool>search</tool><tool_input>旧金山天气</tool_input>
# <observation>64度</observation>
# ```
# 完成后，请在 `<final_answer></final_answer>` 之间回复最终答案。 例如：
# 
# `<final_answer>`旧金山的天气温度为 64 度`</final_answer>`
# 
# 开始！
# 
# 之前的对话：
# 
# 
# Question: 今天广州的天气
# 
# `<tool>weather</tool><tool_input>广州</tool_input><observation>{'description': '阴', 'temperature': '21'}</observation>`
# 
# 

# %%



