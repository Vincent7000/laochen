# ## LM Studio
# 
# 官网链接：https://lmstudio.ai/
# 
# LM Studio 是一款桌面应用程序，专门用于本地部署和运行大型语言模型（LLMs）。这个应用的核心优势在于它极大地降低了运行这些复杂模型的技术门槛，让即使是没有编程基础的普通用户也能够轻松地在本地运行这些模型。
# 主要特点包括：
# 1. **模型选择与下载**：LM Studio 提供了一个用户友好的界面，用户可以直接从中选择和下载多种大型语言模型。这些模型主要托管在 HuggingFace 网站上，包括一些热门的开源模型，例如 Mistral 7B、Codex、Blender Bot、GPT-Neo 等。
# 2. **简单直观的操作流程**：用户只需选择喜欢的模型，点击下载，等待下载完成后，通过 LM Studio 的对话界面加载本地模型，就可以开始与 AI 进行对话。
# 3. **API 转换功能**：LM Studio 还内置了将本地模型快速封装成与 OpenAI 接口兼容的 API 功能。这意味着用户可以将基于 OpenAI 开发的应用程序直接指向本地模型，实现相同的功能，并且完全免费。
# 4. **易用性和兼容性**：LM Studio 的设计考虑到了易用性和兼容性，使得用户可以轻松地在本地与各种高水平的 AI 模型进行交互。
# 5. **本地化运行**：该应用支持在本地运行大语言模型，避免了将数据发送到远程服务器的需要，这对于注重数据隐私和安全的用户来说是一个重要的优势。
# 总的来说，LM Studio 为普通用户提供了便捷的途径来探索和使用大型语言模型，无需复杂的环境配置或编程知识，即可在本地与高级 AI 模型进行交互。
# 
# ![image.png](attachment:image.png)
# 
# ## vLLM
# 官网链接：https://docs.vllm.ai/en/latest/
# 
# vLLM 是由加州大学伯克利分校的 LMSYS 组织开发的一个开源大语言模型高速推理框架。这个框架的主要目的是显著提升语言模型服务在实时场景下的吞吐量和内存使用效率。vLLM 是一个快速且易于使用的库，专门用于大语言模型（LLM）的推理和服务，并且可以与 HuggingFace 无缝集成。
# vLLM 框架的核心特点包括：
# 1. **高性能**：vLLM 在吞吐量方面表现出色，其性能比 Hugging Face Transformers（HF）高出 24 倍，比文本生成推理（TGI）高出 3.5 倍。
# 2. **创新技术**：vLLM 利用了全新的注意力算法「PagedAttention」，有效地管理注意力键和值，从而提高内存使用效率。
# 3. **易用性**：vLLM 的主框架由 Python 实现，便于用户进行断点调试。其系统设计工整规范，结构清晰，便于初学者理解和上手。
# 4. **关键组件**：vLLM 的核心模块包括 LLMEngine、Scheduler、BlockSpaceManager、Worker 和 CacheEngine。这些模块协同工作，实现了高效的推理和内存管理。
# 5. **显存优化**：vLLM 框架通过其创新的显存管理原理，优化了 GPU 和 CPU 内存的使用，从而提高了系统的性能和效率。
# 6. **应用广泛**：vLLM 可用于各种自然语言处理和机器学习任务，如文本生成、机器翻译等，为研究人员和开发者提供了一个强大的工具。
# 综上所述，vLLM 是一个高效、易用且具有创新技术的开源大语言模型推理框架，适用于广泛的自然语言处理和机器学习应用。
# 
# ## API for Open LLMs
# GitHub地址：https://github.com/xusenlinzy/api-for-open-llm
# 
# API for Open LLMs 是一个强大的开源大模型统一后端接口，它提供与 OpenAI 相似的响应。这个接口支持多种开源大模型，如 ChatGLM、Chinese-LLaMA-Alpaca、Phoenix、MOSS 等。它允许用户通过简单的 API 调用来使用这些模型，从而提供了一种便捷的方式来运行和部署大型语言模型。
# API for Open LLMs 的主要特点包括：
# 1. **模型支持**：支持多种流行的开源大模型，用户可以根据需要选择不同的模型。
# 2. **易用性**：提供简单易用的接口，用户可以通过调用这些接口来使用模型的功能，无需关心底层的实现细节。
# 3. **高效稳定**：采用了先进的深度学习技术，具有高效稳定的运行性能，可以快速处理大量的语言任务。
# 4. **功能丰富**：提供包括文本生成、问答、翻译等多种语言处理功能，满足不同场景下的需求。
# 5. **可扩展性**：具有良好的可扩展性，用户可以根据自己的需求对模型进行微调或重新训练，以适应特定的应用场景。
# API for Open LLMs 的使用方法非常简单。用户首先需要注册并登录官网获取 API 密钥，然后通过调用相应的 API 接口来使用所需的功能。例如，进行文本翻译时，用户只需调用翻译功能的 API 接口，传递需要翻译的文本作为输入参数，即可获取翻译后的结果。
# 此外，API for Open LLMs 还支持通过 Docker 启动，用户可以构建 Docker 镜像并启动容器来运行服务。它还提供了本地启动的选项，用户可以在本地安装必要的依赖并运行后端服务。
# 总的来说，API for Open LLMs 是一个功能强大、高效稳定且易于使用的开源大模型接口，适用于各种自然语言处理任务。
# 

from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.7,
)

res = chat.invoke("请问2只兔子有多少条腿？")
print(res)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
output_parser = StrOutputParser()

chain = prompt | chat | output_parser

chain.invoke({"topic": "康师傅绿茶"})

for chunk in chain.stream({"topic": "康师傅绿茶"}):
    print(chunk,end="")




