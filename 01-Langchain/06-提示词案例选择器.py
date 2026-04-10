# ## Select by length
# 
# 此示例选择器根据长度选择要使用的示例。当您担心构建的提示会超过上下文窗口的长度时，这非常有用。对于较长的输入，它将选择较少的示例来包含，而对于较短的输入，它将选择更多的示例。

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# 创建一个反义词的任务示例
examples = [
    {"input": "开心", "output": "伤心"},
    {"input": "高", "output": "矮"},
    {"input": "精力充沛", "output": "没精打采"},
    {"input": "粗", "output": "细"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    # 可供选择的示例。
    examples=examples,
    # PromptTemplate 用于格式化示例。
    example_prompt=example_prompt,
    # 格式化示例的最大长度。
    # 长度由下面的 get_text_length 函数测量。
    max_length=25,
    # 用于获取字符串长度的函数，使用
    # 确定要包含哪些示例。被注释掉是因为
    # 如果未指定，则将其作为默认值提供。
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)
dynamic_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入的反义词",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

#示例输入量较小，因此选择所有示例。
print(dynamic_prompt.format(adjective="big"))

# 示例输入较长，因此仅选择一个示例。
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))

# 您也可以将示例添加到示例选择器。
new_example = {"input": "胖", "output": "瘦"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="热情"))

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

chain = dynamic_prompt | chat | output_parser

res = chain.invoke({"adjective":"热情"})

print(res)

# ## 最大余弦相似度的嵌入示例
# MaxMarginalRelevanceExampleSelector 根据与输入最相似的示例组合来选择示例，同时还针对多样性进行优化。它通过查找与输入具有最大余弦相似度的嵌入示例来实现这一点，然后迭代地添加它们，同时惩罚它们与已选择示例的接近程度。
# 
# ```
# pip install sentence-transformers
# pip install faiss-cpu
# ```

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# embeddings_path = "D:\\ai\\download\\bge-large-zh-v1.5"
# embeddings_path = "/home/vincent/.lmstudio/models/bge-large-zh-v1.5"
embeddings_path = "/home/vincent/.lmstudio/models/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path, model_kwargs={"device": "cuda"})  # 显式指定 cpu、cuda
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_path, model_kwargs={"device": "cpu"})  # 显式指定 cpu、cuda


example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 创建反义词的假装任务的示例。
examples = [
    {"input": "高", "output": "矮"},
    {"input": "精力充沛", "output": "没精打采"},
    {"input": "粗", "output": "细"},
    {"input": "快乐", "output": "悲伤"},
]

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # 可供选择的示例列表。
    examples,
    # 嵌入类用于生成用于测量语义相似性的嵌入。
    embeddings,
    # VectorStore 类用于存储嵌入并进行相似性搜索。
    FAISS,
    # 要生成的示例数量。
    k=2,
)
mmr_prompt = FewShotPromptTemplate(
    # 提供一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入的反义词",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

# 输入是一种感觉，所以应该选择快乐/悲伤的例子作为第一个
print(mmr_prompt.format(adjective="得意"))

chain = mmr_prompt | chat | output_parser

res = chain.invoke({"adjective":"得意"})
print(res)

# ## 通过n-gram重叠选择
# 
# NGramOverlapExampleSelector 根据 ngram 重叠得分，根据与输入最相似的示例来选择示例并对其进行排序。 ngram 重叠分数是 0.0 到 1.0 之间的浮点数（含 0.0 和 1.0）。
# 
# 选择器允许设置阈值分数。 ngram 重叠分数小于或等于阈值的示例被排除。默认情况下，阈值设置为 -1.0，因此不会排除任何示例，只会对它们重新排序。将阈值设置为 0.0 将排除与输入没有 ngram 重叠的示例。

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 虚构翻译任务的示例。.
examples = [
    {"input": "See Spot run.", "output": "请参阅现场运行。"},
    {"input": "My dog barks.", "output": "我的狗吠叫。"},
    {"input": "cat can run", "output": "猫会跑"},
]

example_selector = NGramOverlapExampleSelector(
    # 可供选择的示例。
    examples=examples,
    # PromptTemplate 用于格式化示例。
    example_prompt=example_prompt,
    # 选择器停止的阈值。
    # 默认设置为-1.0。
    threshold=-1.0,
    # 对于负阈值：择器按 ngram 重叠分数对示例进行排序，并且不排除任何示例。
    # 对于大于 1.0 的阈值：选择器排除所有示例，并返回一个空列表。
    # 对于阈值等于 0.0:选择器按 ngram 重叠分数对示例进行排序，并排除那些与输入没有 ngram 重叠的内容。
)
dynamic_prompt = FewShotPromptTemplate(
    # 提供了一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="为每个输入提供中文翻译",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)

# 一个与“cat can run”有大量 ngram 重叠的示例输入。
# 并且与“我的狗吠”没有重叠。
print(dynamic_prompt.format(sentence="cat can run fast."))

# 您可以设置排除示例的阈值。
# 例如，设置阈值等于0.0
# 排除与输入没有 ngram 重叠的示例。
# 自从“我的狗叫了。” 与“cat can run fast”没有 ngram 重叠。
# 它被排除在外。
example_selector.threshold = 0.0
print(dynamic_prompt.format(sentence="cat can run fast."))

# 设置小的非零阈值
example_selector.threshold = 0.09
print(dynamic_prompt.format(sentence="cat can play bird."))

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

chain = dynamic_prompt | chat | output_parser

res = chain.invoke({"sentence":"cat can play bird."})
print(res)



