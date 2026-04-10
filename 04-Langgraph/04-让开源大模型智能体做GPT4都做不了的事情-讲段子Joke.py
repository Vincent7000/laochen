# %%
# 写笑话
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

# %%
from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "lm-studio"
openai_api_base = "http://127.0.0.1:1234/v1"
# llm = ChatOpenAI(
#     openai_api_key=openai_api_key,
#     openai_api_base=openai_api_base,
#     temperature=0.3,
#     model_kwargs = {
#         "frequency_penalty":0.9,
#         "presence_penalty":0.9
#     }
# )
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.3,
    # model="dranger003/UNA-SimpleSmaug-34b-v1beta-iMat.GGUF/ggml-una-simplesmaug-34b-v1beta-q4_k.gguf"
    # model="/mnt/d/ai/download/Smaug-34B-v0.1-AWQ"
)

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """你是1个写段子的能手。请根据下列生成段子的案例的方法，生成段子。
------
1. 主题：讲1个减肥的笑话

2. 关联：找出与减肥有相同特征，但又让人意想不到的对象。
减肥就像戒毒？因为两者都需要毅力、耐心和坚定的决心。不同的是，在戒毒时你可以避开毒品，而在减肥中你却总是被美食诱惑着！

3. 做铺垫：将意想不到对象的某个属性或者特征与减肥的某个属性关联。戒毒跟贩毒相关联，贩毒关押起来。减肥跟卖宵夜的关联。所以买宵夜的也要关押起来。

4. 结合前面内容，将铺垫放到前面，引出意想不到的结果。
段子：我认为卖宵夜的应该被关起来。现在我减肥比戒毒还难。你想想在戒毒时你可以避开毒品，而在减肥中那些卖宵夜的却总是正大光明的诱惑你，还不犯法。
------

根据上面的案例的方法生成段子：

主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

# %%
chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

我是一个很自律的人，既然说了减肥，那就一定会坚持说。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧，并按照这个技巧生成下列主题的段子

主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser


# %%
chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

我是一个很自律的人，既然说了减肥，那就一定会坚持说。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser


# %%
chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

我是一个很自律的人，既然说了减肥，那就一定会坚持说。

这个段子使用了以下技巧：\n\n1. 反转：原本是谈论自律的话题，却以一个出乎意料的结尾来制造笑点。\n2. 双关语："既然说了减肥，那就一定会坚持说"这句话既可以理解为对减肥的决心，也可以理解为对继续谈论减肥话题的坚持。\n3. 幽默感：通过将严肃的主题（自律）与意想不到的结尾相结合，营造出一种荒谬感，从而引发观众的笑声。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser


# %%
chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

每天都吵着要减肥，其实就是想吓吓身上的肉肉。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

每天都吵着要减肥，其实就是想吓吓身上的肉肉。
------
这个段子使用了以下技巧：\n1. 幽默：通过夸张和反转的手法来制造笑点，将减肥的决心与实际行动形成鲜明对比，引发读者共鸣。\n2. 自嘲：以自我调侃的方式表达对减肥的无奈，拉近了与读者的距离，营造轻松氛围。\n3. 简洁明了：句子简短有力，易于理解，符合现代人快节奏的生活习惯。
你是1个写段子的能手。请按照这个技巧生成下列主题的段子

主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser
chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

我有一颗减肥的心和一个吃货的胃，它俩天天掐架，掐指一算，艾玛今天胃又赢了。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

我有一颗减肥的心和一个吃货的胃，它俩天天掐架，掐指一算，艾玛今天胃又赢了。
------
这段话运用了幽默和对比的手法来表达作者内心的矛盾。通过将"减肥的心"与"吃货的胃"拟人化，赋予它们人格化的特点，并让它们进行一场有趣的"掐架"，生动地展现了作者在减肥和美食之间难以抉择的心理状态。同时，用"掐指一算"和"艾玛今天胃又赢了"等口语化的表达方式，增强了段子的亲切感和幽默感，使读者更容易产生共鸣。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

人家减肥减腰减大腿，为什么你非要从脑细胞开始。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

人家减肥减腰减大腿，为什么你非要从脑细胞开始。
------
这段文字的妙处在于运用了反讽的手法，将减肥的关注点由常见的身体部位转移到了大脑（脑细胞）。这种出乎意料、却又让人忍俊不禁的幽默方式，产生了强烈的对比和喜剧效果。同时，以一种调侃的方式提醒人们，真正的减肥应该关注健康饮食和适量运动，而非一味地减少大脑资源。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

我不是不减肥，只是怕反弹罢了。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
我不是不减肥，只是怕反弹罢了。

这个段子使用了以下技巧：
1. 反转：原本看似是关于减肥的讨论，但最后却揭示出主角担心的是反弹而不是减肥本身。这种反转增加了幽默感。\n2. 夸张：将不减肥的原因归结为怕反弹，而非其他更合理的原因，这种夸张手法增强了段子的喜剧效果。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """

谁说吃货除了吃就什么都不会了，她们还知道饿。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
谁说吃货除了吃就什么都不会了，她们还知道饿。

这个段子使用了以下技巧：
这段段子运用了幽默和反转的技巧。首先，它通过"吃货除了吃就什么都不会了"这个普遍的观点来制造预期，然后以"她们还知道饿"作为转折，打破预期并产生幽默效果。这种技巧利用了人们的心理预期，在预期的落差中引发笑点。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
我减肥的时候你一定要来哦，因为看见你，我就没有食欲了。
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
我减肥的时候你一定要来哦，因为看见你，我就没有食欲了。

这个段子使用了以下技巧：
1. 自嘲：通过自嘲自己的身材，引发共鸣或幽默感。\n2. 反转：原本是邀请对方来一起减肥，但最后却说看见对方就没有食欲了，形成了反转效果。\n3. 双关语："没有食欲"既可以理解为减肥时看到食物不想吃，也可以理解为看到某个人就没了兴趣。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
冬天穿个绿色的棉袄，竟然被小朋友说冬瓜会走路了，真气人！
------
你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
冬天穿个绿色的棉袄，竟然被小朋友说冬瓜会走路了，真气人！

这个段子使用了以下技巧：
这个段子运用了以下技巧：\n1. 幽默对比：将穿着绿色棉袄的人与冬瓜进行对比，制造出一种滑稽的效果。\n2. 夸张手法：将穿绿色棉袄的人比喻成会走路的冬瓜，夸大了两者之间的相似性，增强了幽默感。\n3. 反转和讽刺：通过小朋友对穿着绿色棉袄的人的评价，揭示了人们对于不合常规的衣着或行为的看法，引发读者共鸣的同时也产生了喜剧效果。

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

chain.invoke({"input":"讲1个学习的段子"})

# %%
list = [
    "减肥是人生第二大事，第一大事是吃好喝好",
    "冬天穿个绿色的棉袄，竟然被小朋友说冬瓜会走路了，真气人！",
    "抛个硬币，如果黏在天花板上就不吃宵夜了",
    "别问我天天减肥有没有瘦，搞笑，你天天上班有存款吗?",
    "你喊别的女生出去吃饭，她答应了，可能是对你有意思；你喊我出去吃饭 我答应了，那我是真的喜欢吃饭",
    "任何时间，任何地点，超级侦探，认真干饭",
    "我减肥的时候你一定要来哦，因为看见你，我就没有食欲了。",
    "谁说吃货除了吃就什么都不会了，她们还知道饿。",
    "人家减肥减腰减大腿，为什么你非要从脑细胞开始。",
    "我是一个很自律的人，既然说了减肥，那就一定会坚持说。",
    "勇敢是什么，是我明知道这一顿吃下去会胖，我还是迎难而上。",
    "减肥哪有那么容易？我的每块肉都有它的脾气！",
    "我消极的对待减肥，能不能取消我胖子的资格啊！",
    "减肥简直是世界上最反人类的事情，不吃饭饿得想打人，可吃完饭又想打自己。",
    "当一两个人说我胖的时候，我不以为然，后来越来越多的人说我胖，这个时候我终于意识到了事情的严重性，这个世界上的骗子真是越来越多了。",
    "其实我小时候挺瘦的，后来上学了，一句“谁知盘中餐，粒粒皆辛苦”让我变成了如今这副模样。",
    "从来都不用化妆品，我保持年轻的秘诀就是，谎报年龄。",
    "妈妈说不能交不三不四的朋友，所以我的朋友都很二。",
    "做坏事早晚都会被发现，深思熟虑之后，我都改中午做。",
    "你想一夜暴富吗？你想一夜资产过亿吗？不如和我在一起，我们一起想。",
    "别看我平时对你总是漠不关心的样子，其实背底下说了你好多坏话。",
    "没钱的日子来找我，我来告诉你一个馒头，怎么分两天吃？",
    "我每晚都会对自己说：熬夜会死，事实证明我真的不怕死",
    "如果你有喜欢的女生，就送她一支口红吧，至少她亲别人的时候，你还有参与感。",
    "你瘦的时候在我心里，后来胖了，卡在里面出不来了",
]

# %%
from langchain_core.runnables import RunnablePassthrough
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """{input}

你是1个写段子的能手。请分析上面的段子使用了什么技巧
""")
        ]
)

chain = ({"input": RunnablePassthrough()} | assistant_prompt | llm | output_parser)

resultList = chain.batch(list)
resultList

# %%
import json
jsonList = []
for index in range(len(list)):
    jsonList.append({"joke":list[index],"skill":resultList[index]})

print(jsonList)

# 写入JSON数据
with open("jokeData.json", 'w',encoding='utf-8') as f:
    # 确保指定ensure_ascii为False以支持中文字符
    json.dump(jsonList, f, ensure_ascii=False, indent=4)


# %%
import json
# 读取JSON数据
with open("jokeData.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打印加载的数据
print(data)

# %%
import random
# 随机获取n条数据，可能会重复
random_items = random.choices(data, k=5)

# 打印随机获取的数据
print(random_items)

# %%
demoList = []

for item in random_items:
    item["input"] = "讲1个学习的段子"
    demoList.append(item)

# %%
assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """{joke}

这个段子使用了以下技巧：
{skill}

你是1个写段子的能手。请按照上面技巧生成下列主题的段子
主题：{input}
""")
        ]
)

chain = assistant_prompt | llm | output_parser

resList = chain.batch(demoList)
resList

# %%



