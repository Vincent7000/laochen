# %% [markdown]
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
# - Thought 1：“我现在去厨房。”
# - Act 1：“启动进入厨房。”
# - Obs 1：“看到厨房内的环境。”
# 
# - Thought 2：“我需要找到咖啡粉和咖啡机。”
# - Act 2：“寻找咖啡粉和咖啡机。”
# - Obs 2：“找到了咖啡粉和咖啡机。”
# 
# - Thought 3：“我现在可以开始煮咖啡了。”
# - Act 3：“开始煮咖啡。”
# - Obs 3：“咖啡做好了。”
# 
# - Thought 4：“可以把咖啡送过去了。”
# - Act 4：“送咖啡。”
# - Obs 4：“得到了表扬。”
# 
# ReAct能够首先通过推理问题（Thought 1），然后执行一个动作（Act 1）。然后它收到了一个观察（Obs 1），并继续进行这个思想，行动，观察循环，直到达到结论。

# %% [markdown]
# ```python
# PromptTemplate(
#     input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
#     template='Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}'
# )
# ```

# %% [markdown]
# ## 提示词
# 
# ```
# Answer the following questions as best you can. You have access to the following tools:
# 
# {tools}
# 
# Use the following format:
# 
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
# 
# Begin!
# 
# Question: {input}
# Thought:{agent_scratchpad}
# ```
# 
# 
# ```
# 尽可能回答以下问题。 您可以使用以下工具：
# 
# {tools}
# 
# 使用以下格式：
# 
# Question：您必须回答的输入问题
# Thought：你应该时刻思考该做什么
# Action：要采取的操作，应该是 [{tool_names}] 之一
# Action Input：动作的输入
# Observation：行动的结果
# ...（这个想法/行动/行动输入/观察可以重复N次）
# 想法：我现在知道了最终答案
# Final Answer：原始输入问题的最终答案
# 
# 开始！
# 
# 问题：{input}
# 想法：{agent_scratchpad}
# ```

# %%



