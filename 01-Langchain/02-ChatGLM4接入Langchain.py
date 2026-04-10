# ## 智谱AI
# 
# ### GLM-4
# 
# 
# 新一代基座大模型GLM-4，整体性能相比GLM3全面提升60%，逼近GPT-4；支持更长上下文；更强的多模态；支持更快推理速度，更多并发，大大降低推理成本；同时GLM-4增强了智能体能力。
# 
# 
# 基础能力（英文）：GLM-4 在 MMLU、GSM8K、MATH、BBH、HellaSwag、HumanEval等数据集上，分别达到GPT-4 94%、95%、91%、99%、90%、100%的水平。
# 
# ![image.png](https://zhipuaiadmin.aminer.cn/upload/c63e9d2c38338718e79735db348b4a0f.png)
# 
# 
# 指令跟随能力：GLM-4在IFEval的prompt级别上中、英分别达到GPT-4的88%、85%的水平，在Instruction级别上中、英分别达到GPT-4的90%、89%的水平。
# 
# ![image.png](https://zhipuaiadmin.aminer.cn/upload/5125c6da6441a9ce4341ca6c9ccc9b5e.png)
# 
# 
# 对齐能力：GLM-4在中文对齐能力上整体超过GPT-4。
# 
# ![image.png](https://zhipuaiadmin.aminer.cn/upload/e786863b39e483b2282bb94bac61d074.png)
# 
# 
# 长文本能力：我们在LongBench（128K）测试集上对多个模型进行评测，GLM-4性能超过 Claude 2.1；在「大海捞针」（128K）实验中，GLM-4的测试结果为 128K以内全绿，做到100%精准召回。
# 
# ![image.png](https://zhipuaiadmin.aminer.cn/upload/77b81ce140053863d06c5c82b5f0fe18.png)
# 
# 
# 多模态-文生图：CogView3在文生图多个评测指标上，相比DALLE3 约在 91.4% ~99.3%的水平之间。
# 
# ![image.png](https://zhipuaiadmin.aminer.cn/upload/405c46cab17aa3c1da98daa487264723.png)
# 
# 
# 
# 
# GLM-4 实现自主根据用户意图，自动理解、规划复杂指令，自由调用网页浏览器、Code Interpreter代码解释器和多模态文生图大模型，以完成复杂任务。
# 简单来讲，即只需一个指令，GLM-4会自动分析指令，结合上下文选择决定调用合适的工具。
# 
# 
# GLM-4能够通过自动调用python解释器，进行复杂计算（例如复杂方程、微积分等），在GSM8K、MATH、Math23K等多个评测集上都取得了接近或同等GPT-4 All Tools的水平。
# 
# 
# GLM-4 能够自行规划检索任务、自行选择信息源、自行与信息源交互，在准确率上能够达到 78.08，是GPT-4 All Tools 的116%。
# 
# 
# GLM-4 能够根据用户提供的Function描述，自动选择所需 Function并生成参数，以及根据 Function 的返回值生成回复；同时也支持一次输入进行多次 Function 调用，支持包含中文及特殊符号的 Function 名字。这一方面GLM-4 All Tools 与 GPT-4 Turbo 相当。
# 
# 
# 
# 
# 
# 
# 
# 网站链接： https://www.zhipuai.cn/

import os
zhipuai_api_key = os.getenv('ZHIPU_API_KEY')

# pip install --upgrade zhipuai

print(zhipuai_api_key)

from zhipuai import ZhipuAI

client = ZhipuAI(api_key=zhipuai_api_key)

# prompt = "以色列为什么喜欢战争？"
prompt = "为什么有的猫不咬老鼠"
response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role":"user","content":"你好"},
        {"role":"assistant","content":"我是人工智能助手"},
        {"role":"user","content":prompt}
    ]
)



print(response)

print(response.choices[0].message.content)

prompt = "glm-4原理是什么？使用了多少的参数进行训练？"
response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role":"user","content":"你好"},
        {"role":"assistant","content":"我是人工智能助手"},
        {"role":"user","content":prompt}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content,end="")

# GLM-130B（General Language Modeling）是一种大规模的双语预训练语言模型，它的原理基于Transformer架构，这种架构广泛用于自然语言处理任务，因为它能够有效地捕捉文本中的长距离依赖关系。
# 
# ### GLM-130B的原理：
# 1. **Transformer架构**：GLM-130B采用了Transformer模型，该模型基于自注意力机制，能够处理变长序列数据，并在各个层次捕捉文本中的复杂关系。
#    
# 2. **自注意力机制**：模型通过自注意力机制，可以同时考虑输入序列中的所有位置，为每个位置的词分配不同的注意力权重，从而更好地理解上下文。
# 
# 3. **预训练任务**：GLM-130B在预训练阶段使用了多种任务，如掩码语言模型（Masked Language Modeling, MLM）和下一句预测（Next Sentence Prediction, NSP），以提高模型对语言的理解能力。
# 
# 4. **双语能力**：GLM-130B特别强调中英双语能力，这意味着它可以同时处理中文和英文数据，为跨语言任务提供支持。
# 
# 5. **参数高效利用**：尽管参数量巨大，GLM-130B通过有效的训练策略和模型设计，使得参数能够高效地学习和表征语言。
# 
# ### 训练参数量：
# GLM-130B模型的参数量达到了1300亿（130B），这是一个非常庞大的参数规模，使得模型能够捕捉到极其复杂的语言特征和模式。
# 
# 这样的参数规模使得GLM-130B能够处理大规模的数据集，并在多种语言任务上表现出色。同时，为了能够运行在相对实惠的硬件上，GLM-130B还进行了INT4量化，以减少对计算资源的需求，同时尽量保持性能不受损失。
# 
# 总的来说，GLM-130B是一个具有强大表达能力的模型，旨在提供高质量的双语语言理解服务。

from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from zhipuai import ZhipuAI
from langchain_core.messages.ai import AIMessage
import os
from pydantic import Field


class ChatGLM4(LLM):
    # 声明所有模型字段
    client: Any = Field(default=None, exclude=True)  # exclude=True 表示不包含在序列化中
    history: List[Dict[str, str]] = Field(default_factory=list)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        zhipuai_api_key = os.getenv('ZHIPU_API_KEY')
        if not zhipuai_api_key:
            raise ValueError("ZHIPU_API_KEY 环境变量未设置")
        self.client = ZhipuAI(api_key=zhipuai_api_key)
    
    @property
    def _llm_type(self):
        return "ChatGLM4"
    
    def invoke(self, prompt, config={}, history=None):
        if history is None:
            history = []
        if not isinstance(prompt, str):
            prompt = prompt.to_string()
        messages = history + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=messages
        )
        return AIMessage(content=response.choices[0].message.content)
        
    def _call(self, prompt, stop=None, **kwargs):
        return self.invoke(prompt, history=kwargs.get('history', []))
        
    def stream(self, prompt, stop=None, **kwargs):
        history = kwargs.get('history', [])
        if not isinstance(prompt, str):
            prompt = prompt.to_string()
        messages = history + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

llm = ChatGLM4()

llm.invoke("请讲1个减肥的笑话")

for deltaStr in llm.stream("如何鼓励自己减肥！"):
    print(deltaStr,end="")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({"topic": "康师傅绿茶"})

for chunk in chain.stream({"topic": "康师傅绿茶"}):
    print(chunk,end="")



# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)


from langchain_huggingface import HuggingFaceEmbeddings

embeddings_path = "/home/vincent/.lmstudio/models/bge-large-zh-v1.5"
# embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path, model_kwargs={"device": "cuda"})  # 显式指定 cpu、cuda
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path, model_kwargs={"device": "cpu"})  # 显式指定 cpu、cuda


# 两种使用方式：

# 方式1：直接使用模型名称（会自动下载）
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
# embeddings = HuggingFaceEmbeddings(model_name="bge-large-zh-v1.5")

# 方式2：使用本地路径（确保路径正确）
# embeddings_path = "/path/to/your/bge-large-zh-v1.5"  # Linux路径示例
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)


from langchain_core.prompts import ChatPromptTemplate

template ="""
只根据以下文档回答问题：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["小明在华为工作","熊喜欢吃蜂蜜"],
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

from langchain_core.runnables import RunnableParallel,RunnablePassthrough

setup_and_retrieval = RunnableParallel(
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }
)

chain = setup_and_retrieval | prompt | llm

chain.invoke("小明在哪里工作？")

from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://www.vmall.com/help/faq-4359.html", "https://www.vmall.com/help/faq-7923.html"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
docs

from langchain_community.document_transformers import Html2TextTransformer

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

docs_transformed

from langchain_community.document_transformers import BeautifulSoupTransformer

bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    docs, tags_to_extract=["p", "h3"]
)

docs_transformed

from langchain_community.document_loaders import PlaywrightURLLoader
import asyncio

loader = PlaywrightURLLoader(urls=urls, remove_selectors=[".shortcut", ".top-banner"])

data = loader.load()

data




