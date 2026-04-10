# ## LangChain
# 
# LangChain 是一个开源框架，旨在帮助开发者使用大型语言模型（LLMs）和聊天模型构建端到端的应用程序。它提供了一套工具、组件和接口，以简化创建由这些模型支持的应用程序的过程。LangChain 的核心概念包括组件（Components）、链（Chains）、模型输入/输出（Model I/O）、数据连接（Data Connection）、内存（Memory）和代理（Agents）等。
# ![image-2.png](attachment:image-2.png)
# 
# 
# 以下是LangChain的一些关键特性和组件的详细解释：
# 
# 1. **组件（Components）**：
#    - **模型输入/输出（Model I/O）**：负责管理与语言模型的交互，包括输入（提示，Prompts）和格式化输出（输出解析器，Output Parsers）。
#    - **数据连接（Data Connection）**：管理向量数据存储、内容数据获取和转换，以及向量数据查询。
#    - **内存（Memory）**：用于存储和获取对话历史记录的功能模块。
#    - **链（Chains）**：串联Memory、Model I/O和Data Connection，以实现串行化的连续对话和推理流程。
#    - **代理（Agents）**：基于链进一步串联工具，将语言模型的能力和本地、云服务能力结合。
#    - **回调（Callbacks）**：提供了一个回调系统，可连接到请求的各个阶段，便于进行日志记录、追踪等数据导流。
# 
# 2. **模型输入/输出（Model I/O）**：
#    - **LLMs**：与大型语言模型进行接口交互，如OpenAI、Cohere等。
#    - **Chat Models**：聊天模型是语言模型的变体，它们以聊天信息列表为输入和输出，提供更结构化的消息。
# 
# 3. **数据连接（Data Connection）**：
#    - **向量数据存储（Vector Stores）**：用于构建私域知识库。
#    - **内容数据获取（Document Loaders）**：获取内容数据。
#    - **转换（Transformers）**：处理数据转换。
#    - **向量数据查询（Retrievers）**：查询向量数据。
# 
# 4. **内存（Memory）**：
#    - 用于存储对话历史记录，以便在连续对话中保持上下文。
# 
# 5. **链（Chains）**：
#    - 是组合在一起以完成特定任务的一系列组件。
# 
# 6. **代理（Agents）**：
#    - 基于链的工具，结合了语言模型的能力和本地、云服务。
# 
# 7. **回调（Callbacks）**：
#    - 提供了一个系统，可以在请求的不同阶段进行日志记录、追踪等。
# 
# LangChain 的使用场景包括但不限于文档分析和摘要、聊天机器人、代码分析、工作流自动化、自定义搜索等。它允许开发者将语言模型与外部计算和数据源相结合，从而创建出能够理解和生成自然语言的应用程序。
# 
# 要开始使用LangChain，开发者需要导入必要的组件和工具，组合这些组件来创建一个可以理解、处理和响应用户输入的应用程序。LangChain 提供了多种组件，例如个人助理、文档问答、聊天机器人、查询表格数据、与API交互等，以支持特定的用例。
# 
# LangChain 的官方文档提供了详细的指南和教程，帮助开发者了解如何设置和使用这个框架。开发者可以通过这些资源来学习如何构建和部署基于LangChain的应用程序。

# ## LangChain可以构建哪些应用
# ![image.png](attachment:image.png)
# LangChain 作为一个强大的框架，旨在帮助开发者利用大型语言模型（LLMs）构建各种端到端的应用程序。以下是一些可以使用 LangChain 开发的应用类型：
# 
# 1. **聊天机器人（Chatbots）**：
#    - 创建能够与用户进行自然对话的聊天机器人，用于客户服务、娱乐、教育或其他交互式场景。
# 
# 2. **个人助理（Personal Assistants）**：
#    - 开发智能个人助理，帮助用户管理日程、回答问题、执行任务等。
# 
# 3. **文档分析和摘要（Document Analysis and Summarization）**：
#    - 自动分析和总结大量文本数据，提取关键信息，为用户节省阅读时间。
# 
# 4. **内容创作（Content Creation）**：
#    - 利用语言模型生成文章、故事、诗歌、广告文案等创意内容。
# 
# 5. **代码分析和生成（Code Analysis and Generation）**：
#    - 帮助开发者自动生成代码片段，或者提供代码审查和优化建议。
# 
# 6. **工作流自动化（Workflow Automation）**：
#    - 通过自动化处理日常任务和工作流程，提高工作效率。
# 
# 7. **自定义搜索引擎（Custom Search Engines）**：
#    - 结合语言模型的能力，创建能够理解自然语言查询的搜索引擎。
# 
# 8. **教育和学习辅助（Educational and Learning Aids）**：
#    - 开发教育工具，如智能问答系统、学习辅导机器人等，以辅助学习和教学。
# 
# 9. **数据分析和报告（Data Analysis and Reporting）**：
#    - 使用语言模型处理和分析数据，生成易于理解的报告和摘要。
# 
# 10. **语言翻译（Language Translation）**：
#     - 利用语言模型进行实时翻译，支持多语言交流。
# 
# 11. **情感分析（Sentiment Analysis）**：
#     - 分析文本中的情感倾向，用于市场研究、社交媒体监控等。
# 
# 12. **知识库和问答系统（Knowledge Bases and Q&A Systems）**：
#     - 创建能够回答特定领域问题的智能问答系统。
# 
# LangChain 的灵活性和模块化设计使得开发者可以根据特定需求定制和扩展应用程序。通过将语言模型与外部数据源和APIs结合，LangChain 能够支持广泛的应用场景，从而创造出更加智能和用户友好的软件解决方案。

# pip install openai
# pip install langchain-openai

from langchain_openai import ChatOpenAI
import os
OPENAI_API_KEY= os.getenv('OPEN_API_KEY')
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

llm.invoke("中国的首都是哪里？不需要介绍")

# ## 基本示例：提示+模型+输出解析器
# 
# 最基本和常见的用例是将提示模板和模型链接在一起。为了看看这是如何工作的，让我们创建一个接受主题并生成小红书短文的链：
# 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo-1106",openai_api_key=OPENAI_API_KEY)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "康师傅绿茶"})

# ## Prompt 提示
# 
# prompt 是一个 BasePromptTemplate ，这意味着它接受模板变量的字典并生成一个 PromptValue 。 PromptValue 是一个完整提示的包装器，可以传递给 LLM （它接受一个字符串作为输入）或 ChatModel （它接受一个序列作为输入的消息）。它可以与任何一种语言模型类型一起使用，因为它定义了生成 BaseMessage 和生成字符串的逻辑。

prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")

prompt_value = prompt.invoke({"topic": "康师傅绿茶"})
prompt_value

# ## Model 模型
# 
# 然后 PromptValue 被传递给 model 。在本例中，我们的 model 是 ChatModel ，这意味着它将输出 BaseMessage 

message = model.invoke(prompt_value)
message

# ## Output parser 输出解析器
# 
# 将 model 输出传递给 output_parser ，这是一个 BaseOutputParser ，意味着它接受字符串或 BaseMessage 作为输入。 StrOutputParser 特别简单地将任何输入转换为字符串。
# 
# ![image.png](attachment:image.png)

output_parser.invoke(message)

# ## 流式输出
# 
# 如果我们想流式传输结果，我们需要更改我们的函数

for chunk in chain.stream({"topic": "康师傅绿茶"}):
    print(chunk, end="", flush=True)




