# ## 并行化步骤
# 
# RunnableParallel（又名 RunnableMap）可以轻松并行执行多个 Runnable，并将这些 Runnable 的输出作为映射返回。
# 
# RunnableParallel 对于并行运行独立进程也很有用，因为映射中的每个 Runnable 都是并行执行的。

from langchain_openai import ChatOpenAI, OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# 1.1 大纲
outlinePromptTemplate = '''主题：{theme}
如果要根据主题写一篇文章，请列出文章的大纲。'''

outlinePrompt = ChatPromptTemplate.from_template(outlinePromptTemplate)
print(outlinePrompt)


# 2.1 注意事项
tipsPromptTemplate = '''主题：{theme}
如果要根据主题写一篇文章，应该需要注意哪些方面，才能把这篇文章写好。
'''

tipsPrompt = ChatPromptTemplate.from_template(tipsPromptTemplate)
print(tipsPrompt)

# 3. 标题
query = "2025年中国经济走向与运行趋势"

from langchain_core.output_parsers import StrOutputParser
strParser = StrOutputParser()

# 1.2 大纲chain
outlineChain = outlinePrompt | model | strParser
outline = outlineChain.invoke({"theme":query})
print(outline)

# 2.2 注意事项chain
tipsChain = tipsPrompt | model | strParser
tips = tipsChain.invoke({"theme":query})
print(tips)

# 3.1 完整文章模板
articlePromptTemplate = '''主题：{theme}
大纲：
{outline}

注意事项：
{tips}

请根据上面的主题、大纲和注意事项写出丰富的完整文章内容。
'''

articlePrompt = ChatPromptTemplate.from_template(articlePromptTemplate)
print(articlePrompt)

# 3.2 完整文章chain
articleChain = articlePrompt | model | strParser
res = articleChain.invoke({
    "theme":query,
    "outline":outline,
    "tips":tips
})
print(res)

# 4.1 输出不保留原始theme map_chain-1
from langchain_core.runnables import RunnableParallel

map_chain = RunnableParallel(outline=outlineChain, tips=tipsChain)
res = map_chain.invoke({"theme":query})
print(res)

# 4.2 输出要保留原始theme map_chain-2
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

map_chain = RunnableParallel(outline=outlineChain, tips=tipsChain, theme=itemgetter("theme"))
res = map_chain.invoke({"theme":query})
print(res)

# 5.1 allChain-1
allChain = map_chain | articleChain | strParser
res = allChain.invoke({"theme":query})
print(res)

# 5.2 allChain-2
allChain = (
    {
        "outline":outlineChain,
        "tips":tipsChain,
        "theme":itemgetter("theme")
    }
    | articleChain
    | strParser
)
res = allChain.invoke({"theme":query})
print(res)

# 5.3 allChain-3
allChain = (
    {
        "outline":outlineChain,
        "tips":tipsChain,
        "theme":itemgetter("theme")
    }
    | articlePrompt 
    | model 
    | strParser
)
res = allChain.invoke({"theme":query})
print(res)





'''
这三个chain的主要差异在于**结构复杂度**和**组件粒度**：

## 5.1 版本 - 最简洁
```python
allChain = map_chain | articleChain | strParser
```
**结构**：`并行处理 → 文章生成 → 解析`
- 使用预定义的 `map_chain`（包含并行处理）
- `articleChain` 可能是一个复合链（包含prompt+model）
- **最简洁**，但灵活性较低

## 5.2 版本 - 显式并行结构
```python
allChain = (
    {
        "outline":outlineChain,
        "tips":tipsChain,
        "theme":itemgetter("theme")
    }
    | articleChain
    | strParser
)
```
**结构**：`内联并行字典 → 文章生成 → 解析`
- 将并行处理**内联**在链定义中
- 避免了额外的 `map_chain` 变量
- `articleChain` 仍然是复合链
- **中等复杂度**，更清晰的流程

## 5.3 版本 - 最细粒度控制
```python
allChain = (
    {
        "outline":outlineChain,
        "tips":tipsChain,
        "theme":itemgetter("theme")
    }
    | articlePrompt 
    | model 
    | strParser
)
```
**结构**：`并行字典 → 提示词模板 → 模型 → 解析`
- **完全拆解**了 `articleChain` 的组件
- 明确显示了 `articlePrompt` 和 `model` 的单独步骤
- **最细粒度**的控制，灵活性最高
- 可以单独调整 prompt 或 model

## 关键差异对比

| 特性 | 5.1 版本 | 5.2 版本 | 5.3 版本 |
|------|----------|----------|----------|
| 结构复杂度 | 低 | 中 | 高 |
| 组件粒度 | 粗 | 中 | 细 |
| 灵活性 | 低 | 中 | 高 |
| 可读性 | 高 | 中 | 中 |
| 并行处理方式 | 外部变量 | 内联字典 | 内联字典 |
| articleChain 处理 | 复合链 | 复合链 | 拆解为 prompt+model |

## 等效关系

实际上这三个链在功能上是等价的：

- `articleChain` ≈ `articlePrompt | model`
- `map_chain` ≈ `{"outline":..., "tips":..., "theme":...}`

## 选择建议

- **5.1**：适合快速开发，代码简洁
- **5.2**：平衡可读性和灵活性  
- **5.3**：需要精细控制 prompt 或 model 时使用

## 执行流程对比

**5.1/5.2**：
```
输入 → 并行处理 → articleChain(黑盒) → 解析 → 输出
```

**5.3**：
```
输入 → 并行处理 → 提示词模板 → 大模型 → 解析 → 输出
```

'''