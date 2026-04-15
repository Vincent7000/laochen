# %% [markdown]
# ##  Python REPL
# 
# Python REPL（Read-Eval-Print Loop）是一个交互式编程环境，允许程序员立即执行代码并获取结果。REPL 这个术语来源于描述这种环境工作方式的四个步骤：
# 
# - Read（读取） - 读取用户输入的代码。
# - Eval（求值） - 对输入的代码进行求值。
# - Print（打印） - 打印求值的结果。
# - Loop（循环） - 返回第一步，等待用户输入更多的代码。
# 
# 在 Python 中，REPL 通常是通过命令行工具 python 或 python3（取决于安装的 Python 版本）访问的。启动这个工具后，用户可以直接输入 Python 代码，并立即看到代码执行的结果。
# 

# %%
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from utils import llm, searxng_search


python_repl = PythonREPL()

# %%
python_repl.run("print(1+3)")

# %%
from langchain_openai import ChatOpenAI, OpenAI



# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

promptFormat = """{query}

请根据上面的问题，生成Python代码计算出问题的答案，最后计算出来的结果用print()打印出来，请直接返回Python代码，不要返回其他任何内容的字符串
"""
prompt = ChatPromptTemplate.from_template(promptFormat)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# %%
result = chain.invoke({"query":"3箱苹果重45千克。一箱梨比一箱苹果多5千克，3箱梨重多少千克？"})
print(f"chain1: \n{result.strip()}")

# %%
def parsePython(codeStr):
    codeStr = result.replace("```python","")
    codeStr = result.replace("```","")
    return codeStr

# %%
chain = prompt | llm | output_parser | parsePython
# chain

# %%
result = chain.invoke({"query":"3箱苹果重45千克。一箱梨比一箱苹果多5千克，3箱梨重多少千克？"})
print(f"\nchain2: \n{result.strip()}")

# %%
print(f"chain2 result: {python_repl.run(result)}")

# %%
chain = prompt | llm | output_parser | parsePython | python_repl.run
# chain

# %%
result = chain.invoke({"query":"3箱苹果重45千克。一箱梨比一箱苹果多5千克，3箱梨重多少千克？"})
print(f'chain3 result: {result}')

# %%



