# %% [markdown]
# ![image.png](attachment:d0f8da76-659e-479b-bb48-4214ce142b7c.png)

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key = "empty",
    openai_api_base="http://127.0.0.1:1234/v1",
    temperature=0.3
)

# %%
#定义嘉宾智能体的提示词头部
player_prompt_header = """
请永远记住您现在扮演{agent_role}的角色。

您的基本介绍：{agent_description}
您的性格：{agent_nature}
您的经历：{agent_experience}

目前轮到你发言，请您根据上面的节目聊天内容以及你的角色和经历，以及所处的位置角度提供该主题最丰富、最有创意和最新颖的观点，只返回你要发表的内容。
"""



# %%
# 成龙、刘亦菲、沈腾、董成鹏
roleList = ["成龙","刘亦菲","沈腾","董成鹏"]

# %%
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
strParser = StrOutputParser()
roleDesPrompt = PromptTemplate.from_template("""
用户输入：{input}
请根据用户输入的明星，生成明星的详细介绍。返回必须按照下面的JSON格式返回,只返回json内容，不要返回斜杠注释的说明。
{{
    name:str, //明星的名称
    description: str, //明星的基本介绍
    nature: str, //明星的性格
    experience: str, //明星的经历
}}
""")

roleDesChain = roleDesPrompt | llm | strParser

# %%
from langchain_core.output_parsers.json import parse_json_markdown
batchInput = []
for item in roleList:
    batchInput.append({
        "input":item
    })

roleDesList = roleDesChain.batch(batchInput)
roleDesList

# %%
from langchain_core.output_parsers.json import parse_json_markdown

roleDesListJson = []
for item in roleDesList:
    roleDesListJson.append(parse_json_markdown(item))

roleDesListJson

# %%
topic= "出身-家世决定你多少"

player_prompt = """
这是圆桌派综艺节目，目前讨论以下主题：{topic}

本期节目嘉宾介绍：{roleList}

节目聊天内容：
{chatList}

{roleDesc}
"""

# %%
host_prompt = """
这是圆桌派综艺节目，目前讨论以下主题：{topic}

本期节目嘉宾介绍：
{roleDescList}

节目聊天内容：
{chatList}

下一位发言的嘉宾：{player}

请永远记住您现在扮演节目主持人的角色，你的名字叫陈鹏。
目前轮到你发言，你需要根据上面节目聊天内容的进展来主持节目进行发言。如果节目尚未开始，你需要介绍嘉宾和做本期节目的开场介绍，并引导下一位嘉宾发言,如果没有下一位嘉宾，请做好本次节目的总结并结束节目,只返回发言内容,不要添加其他内容。
"""



# %%


# %%
playersPrompt = []

for role in roleDesListJson:
    prompt = player_prompt_header.format(
        agent_role=role["name"],
        agent_description=role["description"],
        agent_nature=role["nature"],
        agent_experience=role["experience"]
    ) 
    playersPrompt.append(prompt)

playersPrompt

# %%
playerPromptList = []

for item in playersPrompt:
    playerPrompt = PromptTemplate.from_template(player_prompt)
    playerPrompt = playerPrompt.partial(
        roleList=",".join(roleList),
        roleDesc=item
    )
    playerPromptList.append(playerPrompt)

playerPromptList

# %%
playerChains = []

for prompt in playerPromptList:
    chain = prompt | llm 
    playerChains.append(chain)

playerChains

# %%
prompt = PromptTemplate.from_template(host_prompt)
hostChain = prompt | llm 
hostChain

# %%
roleDescListStr = ""
for item in roleDesListJson:
    roleDescListStr = roleDescListStr + item["name"]+":"+item["description"] + "\n"

roleDescListStr

# %%
hostChain.invoke({
    "topic":topic,
    "chatList":"节目刚开始，暂无聊天内容",
    "roleDescList":roleDescListStr,
    "player":"成龙"
})

# %%
from langgraph.graph import MessageGraph,END
import random

graphBuilder = MessageGraph()
data = {
    "topic":topic,
    "chatList":"节目刚开始，暂无聊天内容",
    "roleDescList":roleDescListStr,
    "player":"成龙",
    "isEnd":False
}

def choose(state):
    # print("choose----------")
    # print(state)
    # print(data)
    if data["isEnd"]:
        data["chatList"].append("主持人（陈鹏）："+state[-1].content)
        return "end"
    if len(state) > 5:
        data["isEnd"] = True;
    for index in range(len(roleList)):
        if data["player"] == roleList[index]:
            return "play" + str(index+1)
    return "end"

def msgParser(state):
    # print("msgParser----------")
    # print(state)
    if not isinstance(data["chatList"],str):
        data["chatList"].append("嘉宾（"+data["player"]+"）："+state[-1].content)

    if data["isEnd"]:
        data["player"] = "节目即将结束，不需要下一位嘉宾发言"
        print("节目即将结束，不需要下一位嘉宾发言")
        print(data)
    else:
        # 随机选择一个嘉宾
        random_items = random.choices(roleList,k=1)
        data["player"] = random_items[0]
    return data

def playMsgParser(state):
    # print("playMsgParser----------")
    # print(state)
    if isinstance(data["chatList"],str):
        data["chatList"] = ["主持人（陈鹏）："+state[0].content]
    else:
        data["chatList"].append("主持人（陈鹏）："+state[-1].content)
    return {
        'chatList':data["chatList"],
        'topic':data["topic"]
    }
    

graphBuilder.add_node("hostNode", msgParser | hostChain)
graphBuilder.add_node("playNode1",playMsgParser | playerChains[0])
graphBuilder.add_node("playNode2",playMsgParser | playerChains[1])
graphBuilder.add_node("playNode3",playMsgParser | playerChains[2])
graphBuilder.add_node("playNode4",playMsgParser | playerChains[3])
graphBuilder.add_conditional_edges("hostNode",choose,{
    "end":END,
    "play1":"playNode1",
    "play2":"playNode2",
    "play3":"playNode3",
    "play4":"playNode4",
})
graphBuilder.add_edge("playNode1","hostNode")
graphBuilder.add_edge("playNode2","hostNode")
graphBuilder.add_edge("playNode3","hostNode")
graphBuilder.add_edge("playNode4","hostNode")

graphBuilder.set_entry_point("hostNode")

graph = graphBuilder.compile()

# %%
from IPython.display import Image

Image(graph.get_graph().draw_png())

# %%
graph.invoke([])

# %%


# %%
len(data["chatList"])

# %%
data["chatList"]

# %%



