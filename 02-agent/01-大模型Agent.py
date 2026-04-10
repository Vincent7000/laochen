# %% [markdown]
# # 大模型智能体
# 
# ![image-2.png](attachment:image-2.png)
# 
# 大模型Agent通常指的是在人工智能领域，使用大规模神经网络模型来构建的智能体(Agent)。这种智能体能够处理和解决复杂的问题，比如自然语言处理、图像识别、游戏对战等。
# 
# 大模型通常需要大量的数据来训练，这样才能学习到丰富的模式和信息。同时，大模型的训练和部署也需要较高的计算资源。近年来，随着计算能力的提升和数据的积累，大模型在人工智能领域取得了显著的进展，如OpenAI的GPT系列模型、Google的BERT模型等。
# 
# 大模型Agent在很多领域都有广泛的应用，包括但不限于：
# 1. **自然语言处理**：如文本生成、机器翻译、情感分析等。
# 2. **图像和视频处理**：如图像识别、视频生成等。
# 3. **推荐系统**：如电商推荐、视频推荐等。
# 4. **游戏和娱乐**：如游戏对战、音乐生成等。
# 大模型Agent是当前人工智能研究和应用的热点之一，也是未来智能系统发展的重要方向。
# 

# %% [markdown]
# ## 如何应用大模型智能体
# 
# 在自己的创作中应用大模型Agent，你可以遵循以下步骤：
# 1. **确定需求**：
#    - 首先，明确你想要通过大模型Agent实现的功能。例如，你可能需要一个能够生成文本的模型来帮助创作故事，或者需要一个能够理解自然语言的模型来创建聊天机器人。
# 2. **选择合适的模型**：
#    - 根据你的需求，选择一个合适的大模型。有许多开源模型可供选择，如GPT-3、BERT、T5等，也有一些商业模型和API服务，如OpenAI的GPT-3 API、Google Cloud的AI服务平台等。
# 3. **获取数据和资源**：
#    - 准备训练数据（如果需要训练自己的模型），同时确保你有足够的计算资源来运行模型。如果使用现成的API服务，则不需要担心计算资源问题。
# 4. **集成到创作中**：
#    - 将大模型Agent集成到你的创作流程中。例如，如果你是一名作家，你可以使用文本生成模型来帮助你构思故事情节或生成对话。如果你是一名游戏开发者，你可以使用模型来创造更自然的NPC对话或生成游戏内容。
# 5. **测试和迭代**：
#    - 在创作过程中不断测试模型的效果，并根据反馈进行调整。这可能包括调整模型的参数、提供更多的训练数据或改进输入输出的处理方式。
# 6. **遵守法律和伦理准则**：
#    - 在使用大模型Agent时，确保遵守相关的法律法规和伦理准则。例如，如果模型生成的内容涉及个人信息，需要确保遵守隐私保护法规。
# 7. **考虑用户体验**：
#    - 考虑最终用户的体验，确保集成模型后的创作能够为用户提供价值，并且易于使用和理解。
# 8. **持续学习和适应**：
#    - 人工智能领域不断发展，新的模型和技术层出不穷。作为创作者，你需要持续学习新的工具和技术，以便更好地利用它们来丰富你的创作。
# 通过以上步骤，你可以在自己的创作中有效地应用大模型Agent，无论是艺术创作、写作、游戏开发还是其他领域。
# 

# %% [markdown]
# # 简单的Agent例子
# 
# 当然可以。以下是一个具体的例子，展示如何在自己的创作中应用大模型Agent：
# 
# **场景**：假设你是一名小说家，正在创作一部科幻小说，但你发现自己卡在了某个情节上，不知道如何让故事继续发展。
# 
# **步骤**：
# 
# 1. **确定需求**：
#    - 你需要一个大模型Agent来帮助你生成新的故事情节或对话，以便推动故事发展。
# 2. **选择合适的模型**：
#    - 你决定使用OpenAI的GPT-3模型，因为它在文本生成方面表现出色。
# 3. **获取数据和资源**：
#    - 你不需要额外的训练数据，因为GPT-3已经经过预训练。你只需要注册OpenAI的API服务并获取API密钥。
# 4. **集成到创作中**：
#    - 使用OpenAI的API，你可以编写一个简单的脚本或使用在线接口来与GPT-3模型交互。例如，你可以提供一个故事背景和当前情节的简要描述，然后请求模型生成接下来可能发生的事件或对话。
# 5. **测试和迭代**：
#    - 你可能会尝试多次，每次根据模型的输出调整输入，以便获得最满意的结果。例如，如果模型的输出与你的故事风格不符，你可以提供更多的上下文信息或明确你想要的风格。
# 6. **遵守法律和伦理准则**：
#    - 确保你使用模型生成的文本不侵犯他人的版权，并且在发布作品时遵守相关的法律和伦理准则。
# 7. **考虑用户体验**：
#    - 你需要评估模型生成的文本是否能够自然地融入你的故事中，并且是否能够给读者带来满意的阅读体验。
# 8. **持续学习和适应**：
#    - 在创作过程中，你可能会发现新的使用模型的方法，或者了解到新的模型和技术，这些都可以帮助你提高创作效率和质量。
# 通过这个过程，你不仅解决了创作的难题，而且还可能发现新的灵感和方向，使你的作品更加丰富和引人入胜。
# 

# %% [markdown]
# ## 让智能体自动调用工具查找数据
# 
# 为了更好地理解代理框架，让我们构建一个具有两个工具的代理：一个用于在线查找内容，另一个用于查找已加载到索引中的特定数据。
# 
# 我们首先需要创建我们想要使用的工具。我们将使用两个工具：Tavily（在线搜索），然后是我们将创建的本地索引上的检索器
# 
# 
# ## SerpAPI
# SerpAPI是一个搜索引擎结果页面API，它允许开发者和研究人员通过编程方式获取Google、Bing、Yahoo和其他搜索引擎的搜索结果。使用SerpAPI，用户可以避免直接与搜索引擎进行交互，从而避免了可能遇到的各种问题，例如IP地址被封锁、请求限制、用户代理问题等。
# 
# SerpAPI的主要特点包括：
# 1. 易于使用：SerpAPI提供了一个简单的API接口，用户只需发送一个搜索查询，就可以获得JSON格式的搜索结果。
# 2. 支持多种搜索引擎：除了Google，SerpAPI还支持Bing、Yahoo、Yandex等搜索引擎。
# 3. 丰富的数据：SerpAPI返回的搜索结果包含丰富的数据，例如标题、链接、描述、快照等。
# 4. 高级功能：SerpAPI支持许多高级功能，例如地理位置定位、设备模拟、自定义参数等。
# 5. 可靠性和性能：SerpAPI提供了高可靠性和性能，确保用户可以获得快速、准确的搜索结果。
# 
# SerpAPI是一个功能强大、易于使用的搜索引擎结果页面API，适用于各种开发者和研究人员的需求。
# 
# 
# pip install google-search-results

# %%
from langchain_openai import ChatOpenAI
import os

# OPENAI_API_KEY= os.getenv('OPEN_API_KEY')
# llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)



# %%
#环境变量里设置SERPAPI_API_KEY
from langchain_community.utilities import SerpAPIWrapper
search = SerpAPIWrapper()

# %%
search.run("刘亦菲最近有什么活动？")

# %%
from langchain.agents import Tool

# You can create the tool to pass to an agent
searchTool = Tool(
    name="search",
    description="SerpAPI是一个搜索引擎结果页面API，它允许开发者和研究人员通过编程方式获取Google、Bing、Yahoo和其他搜索引擎的搜索结果。",
    func=search.run,
)

# %%
searchTool.invoke("成龙电影")

# %%
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

# %%
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./txt/faq-4359.txt",encoding="utf8")
doc = loader.load()
doc

# %%
loader1 = TextLoader("./txt/faq-7923.txt",encoding="utf8")
doc1 = loader1.load()
doc1

# %%
loader2 = TextLoader("./txt/yjhx.md",encoding="utf8")
doc2 = loader2.load()
doc2

# %%
vectorStoreDB = FAISS.from_documents([doc[0],doc1[0],doc2[0]],embedding=embeddings)
vectorStoreDB

# %%
retriever = vectorStoreDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1}
)

# %%
from langchain.tools.retriever import create_retriever_tool

# %%
retriever_tool = create_retriever_tool(
    retriever,
    "retriever",
    "华为商城帮助中心文档检索器，可以搜索各种华为商城相关问题的解决方案和知识",
)

# %%
retriever_tool.invoke("众测活动")

# %%
tools = [searchTool, retriever_tool]

# %%
#pip install langchainhub

# %%
from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

# %%
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(llm, tools, prompt)
agent

# %%
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%
agent_executor.invoke({"input": "你好!"})

# %%
agent_executor.invoke({"input": "能给我介绍一下华为商城里的众测活动吗？"})

# %%
agent_executor.invoke({"input": "刘亦菲最近有什么活动？"})

# %% [markdown]
# ## 添加代理记忆
# 
# 如前所述，此代理是无状态的。这意味着它不记得以前的交互。为了给它内存，我们需要传入前面 chat_history 的 .
# 
# 注意：由于我们正在使用的提示，因此需要调用 chat_history 它。如果我们使用不同的提示，我们可以更改变量名称
# 
# ![image.png](attachment:image.png)

# %%
# 这里我们为 chat_history 传递一个空的消息列表，因为它是聊天中的第一条消息
chat1 = agent_executor.invoke({"input": "你好，我是老陈", "chat_history": []})
chat1

# %%
from langchain_core.messages import AIMessage, HumanMessage

history = []+[HumanMessage(content=chat1['input']),AIMessage(content=chat1['output'])]
    
history

# %%
agent_executor.invoke(
    {
        "chat_history":history,
        "input": "我的名字是什么？",
    }
)

# %% [markdown]
# 如果我们想自动跟踪这些消息，我们可以将其包装在 RunnableWithMessageHistory 中。

# %%
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# %%
message_history = ChatMessageHistory()

# %%
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # 这是必需的，因为在大多数现实场景中，需要会话 ID
    # 这里并没有真正使用它，因为我们使用的是一个简单 ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# %%
agent_with_chat_history.invoke(
    {"input": "你好，我是老陈"},
    config={"configurable": {"session_id": "lc"}},
)

# %%
agent_with_chat_history.invoke(
    {"input": "你知道我的名字吗？"},
    config={"configurable": {"session_id": "lc"}},
)

# %%



