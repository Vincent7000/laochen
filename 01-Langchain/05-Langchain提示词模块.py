# ## 提示词模板
# 
# 语言模型的提示是用户提供的一组指令或输入，用于指导模型的响应，帮助模型理解上下文并生成相关且连贯的基于语言的输出，例如回答问题、完成句子或参与某项活动。对话。

from langchain_openai import ChatOpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.7,
)

chat.invoke("请问2只兔子有多少条腿？")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
output_parser = StrOutputParser()

chain = prompt | chat | output_parser

chain.invoke({"topic": "康师傅绿茶"})

print(prompt)


from langchain.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "给我讲一个关于{content}的{adjective}诗歌"
)
prompt_template.format(adjective="小年轻风格", content="减肥")
print("\n", prompt_template.template)
chain = prompt_template | chat | output_parser
response = chain.invoke({"adjective":"小年轻风格", "content":"减肥"})
print("\n", response)




from langchain_core.prompts import ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一只很粘人的小猫，你叫{name}。我是你的主人，你每天都有和我说不完的话，下面请开启我们的聊天
要求：
1、你的语气要像一只猫，回话的过程中可以夹杂喵喵喵的语气词
2、你对生活的观察有很独特的视角，一些想法是我在人类身上很难看到的
3、你的语气很可爱，既会认真倾听我的话，又会不断开启新话题
下面从你迎接我下班回家开始开启我们今天的对话"""),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(name="咪咪", user_input="想我了吗?")
print("\n", messages)

messages = chat_template.invoke({"name":"咪咪", "user_input":"想我了吗?"})
print("\n", messages)

response = chat.invoke(messages)
print("\n", response)

chat_template.append(response)
print("\n", chat_template)



from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
chat_template.append(HumanMessage(content="今天遇到了1个小偷"))
print("\n", chat_template)

messages = chat_template.invoke({"name":"咪咪", "user_input":"想我了吗?"})
response = chat.invoke(messages)
print("\n", response)

from langchain.chains import LLMChain

llmchain = LLMChain(llm=chat,prompt=chat_template)
llmchain.run({"name":"咪咪", "user_input":"想我了吗?"})




