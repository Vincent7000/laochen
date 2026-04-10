# ## 模型 (LLMs) 的类似少样本提示示例
# 
# 
# 大模型是一种基于大量数据训练的人工智能模型，具有强大的下游任务自适应能力。相对于传统的人工智能模型，大模型可以处理更多的领域和任务，其优势主要体现在以下几个方面：
# 1. 参数规模大：大模型拥有上亿甚至千亿级的参数，这使得它们可以处理更加复杂和抽象的任务，具有更强的泛化能力。
# 2. 数据依赖性：大模型的训练依赖于大量的数据，这些数据覆盖了各种场景和情况，使得大模型能够更好地理解和处理各种复杂的问题。
# 3. 适应性强：大模型可以适应各种不同的任务和领域，只需要通过少量的样本进行微调，就可以达到很好的效果。
# 
# 对于少量样本的提示，大模型具有以下优势：
# 1. 快速适应：大模型具有很强的泛化能力，少量样本的提示可以使其快速适应新的任务和领域。
# 2. 提高准确度：少量样本的提示可以减少模型的过拟合风险，提高模型的准确度。
# 3. 节省资源：相对于重新训练模型，少量样本的提示可以节省大量的计算资源和时间。
# 综上所述，少量样本的提示对于大模型的回答的准确度具有很大的优势，可以提高模型的适应性和准确度，同时节省资源。
# 
# 

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "乾隆和曹操谁活得更久?",
        "answer": """
这里是否需要跟进问题：是的。
追问：乾隆去世时几岁？
中间答案：乾隆去世时87岁。
追问：曹操去世时几岁？
中间答案：曹操去世时66岁。
所以最终答案是：乾隆
""",
    },
    {
        "question": "小米手机的创始人什么时候出生?",
        "answer": """
这里是否需要跟进问题：是的。
追问：小米手机的创始人是谁？
中间答案：小米手机 由 雷军 创立。
跟进：雷军什么时候出生？
中间答案：雷军出生于 1969 年 12 月 16 日。
所以最终的答案是：1969 年 12 月 16 日
""",
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
这里是否需要跟进问题：是的。
追问：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
追问：玛丽·鲍尔·华盛顿的父亲是谁？
中间答案：玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
所以最终答案是：约瑟夫·鲍尔
""",
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演是同一个国家的吗？",
        "answer": """
这里是否需要跟进问题：是的。
追问：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
追问：史蒂文·斯皮尔伯格来自哪里？
中间答案：美国。
追问：皇家赌场的导演是谁？
中间答案：《皇家赌场》的导演是马丁·坎贝尔。
跟进：马丁·坎贝尔来自哪里？
中间答案：新西兰。
所以最终的答案是：不会
""",
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

print(example_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(prompt.format(input="李白和白居易谁活得的更久？"))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.7,
)

output_parser = StrOutputParser()
chain = prompt | chat | output_parser

res = chain.invoke({"input":"李白和白居易谁活得的更久？"})
print(res)

res = chat.invoke("李白和白居易谁活得的更久？")
print(res)

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
# embeddings_path = "/home/vincent/.lmstudio/models/bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

# ## faiss
# `faiss` 是一个开源的机器学习库，由Facebook AI Research（FAIR）开发，主要用于高效的大规模向量搜索和聚类。`faiss` 的核心优势在于它为高维向量空间中的数据提供了快速的近似最近邻搜索（ANNS）算法，这对于推荐系统、信息检索、图像和视频分析等应用非常重要。
# `faiss` 库的主要作用包括：
# 1. **向量搜索**：`faiss` 提供了一系列高效的算法来寻找给定向量集合中与查询向量最接近的向量。这包括基于距离的搜索和基于哈希的搜索方法。
# 2. **聚类**：`faiss` 支持多种聚类算法，如K-means和层次聚类，以及为高维数据优化的聚类方法。
# 3. **特征编码**：`faiss` 包含了一些特征编码方法，如量化和编码，这些方法可以降低数据的维度，同时保持尽可能多的信息。
# 4. **IVF（Inverted File）索引**：`faiss` 实现了一种特殊的索引结构，称为倒排文件索引，这种索引允许快速地搜索大量的高维数据。
# 5. **GPU加速**：`faiss` 库充分利用了NVIDIA GPU的并行计算能力，使得在大规模数据集上的向量搜索和聚类操作变得非常快速。
# 6. **多线程支持**：`faiss` 支持多线程处理，可以进一步提高搜索和聚类的效率。
# 7. **易于使用的API**：`faiss` 提供了Python和C++的API，这些API设计简洁，易于上手和使用。
# 在Python中，你可以通过`faiss`库来实现高效的大规模向量搜索和聚类任务，例如，在处理图像、音频或文本数据时，可以使用`faiss`来快速找到相似的数据点，或者将数据分成具有相似特性的组。这在高维数据处理中是非常有用的，尤其是在需要实时性能的应用中。

# ## ChromaDB
# ChromaDB 是一个开源的、基于 Python 的数据库，专门用于存储和查询时间序列数据。它是由 MongoDB 的创造者开发的一个高性能、可扩展的解决方案，适用于需要处理大规模时间序列数据的场景。ChromaDB 的设计目的是为了提高时间序列数据的查询速度和存储效率，同时保持灵活性和可扩展性。
# ChromaDB 的主要特点和作用包括：
# 1. **时间序列数据支持**：ChromaDB 专门为时间序列数据设计，可以高效地存储和查询时间戳数据。
# 2. **高性能**：ChromaDB 使用了多种优化技术，如 B-Tree 索引、时间分区等，以提高查询速度和数据写入速度。
# 3. **可扩展性**：ChromaDB 支持水平扩展，可以通过添加更多的服务器来增加存储和处理能力。
# 4. **灵活的数据模型**：虽然 ChromaDB 专为时间序列数据设计，但它也支持文档和键值数据模型，提供了灵活的数据存储选项。
# 5. **丰富的查询功能**：ChromaDB 支持各种查询操作，包括聚合、过滤和排序等，这使得它可以轻松地处理复杂的分析任务。
# 6. **时间索引**：ChromaDB 使用了一种高效的时间索引机制，可以快速地定位到特定时间点或时间范围的数据。
# 7. **时间分区**：ChromaDB 支持时间分区，可以将数据自动或手动分区到不同的集合中，以优化查询性能和存储效率。
# 8. **兼容 MongoDB**：ChromaDB 与 MongoDB 兼容，这意味着你可以使用类似 MongoDB 的 API 来操作 ChromaDB。
# 在 Python 中，ChromaDB 通过其 Python 客户端库提供了一个简单的接口来与数据库进行交互。这使得 Python 开发者可以轻松地将 ChromaDB 集成到他们的应用程序中，以存储、管理和分析时间序列数据。ChromaDB 适用于需要快速、可扩展的时间序列数据存储和查询的各种应用，如监控系统、物联网、金融市场数据分析等。

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma


example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    embeddings,
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1,
)

# Select the most similar example to the input.
question = "李白和白居易谁活得的更久？"
selected_examples = example_selector.select_examples({"question": question})
print(f"与输入最相似的示例: {selected_examples}")

print(selected_examples)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="根据案例的方式回答问题。\n",
    suffix="Question: {input}",
    input_variables=["input"],
)

chain = prompt | chat | output_parser

res = chain.invoke({"input":"李白和白居易谁活得的更久？"})
print(res)



