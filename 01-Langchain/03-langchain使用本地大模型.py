# ## 自定义调用本地大模型方法
# 1. **类属性定义**:
#    - `max_token`: 定义了模型可以处理的最大令牌数。
#    - `do_sample`: 指定是否在生成文本时采用采样策略。
#    - `temperature`: 控制生成文本的随机性，较高的值会产生更多随机性。
#    - `top_p`: 一种替代`temperature`的采样策略，这里设置为0.0，意味着不使用。
#    - `tokenizer`: 分词器，用于将文本转换为模型可以理解的令牌。
#    - `model`: 存储加载的模型对象。
#    - `history`: 存储对话历史。
# 2. **构造函数**:
#    - `__init__`: 构造函数初始化了父类的属性。
# 3. **属性方法**:
#    - `_llm_type`: 返回模型的类型，即`ChatGLM3`。
# 4. **加载模型的方法**:
#    - `load_model`: 此方法用于加载模型和分词器。它首先尝试从指定的路径加载分词器，然后加载模型，并将模型设置为评估模式。这里的模型和分词器是从Hugging Face的`transformers`库中加载的。
# 5. **调用方法**:
#    - `_call`: 一个内部方法，用于调用模型。它被设计为可以被子类覆盖。
#    - `invoke`: 这个方法使用模型进行聊天。它接受一个提示和一个历史记录，并返回模型的回复和更新后的历史记录。这里使用了模型的方法`chat`来生成回复，并设置了采样、最大长度和温度等参数。
# 6. **流式方法**:
#    - `stream`: 这个方法允许模型逐步返回回复，而不是一次性返回所有内容。这对于长回复或者需要实时显示回复的场景很有用。它通过模型的方法`stream_chat`实现，并逐块返回回复。
# 

from IPython.display import HTML
HTML("<style>div.output_area pre {white-space: pre-wrap;}</style>")

from typing import ClassVar, List, Dict, Any, Optional
from langchain.llms.base import LLM
from transformers import AutoTokenizer,AutoModel,AutoConfig
from langchain_core.messages.ai import AIMessage
from pydantic import Field

class ChatGLM3(LLM):
    max_token:int=8192
    do_sample:bool = True
    temperature:float = 0.3
    top_p:float = 0.0
    tokenizer:object = None
    model:object = None
    # history = []
    history: List[Dict[str, str]] = Field(default_factory=list)  # 添加了类型注解
    
    def __init__(self):
        super().__init__()
    
    @property
    def _llm_type(self):
        return "ChatGLM3"
    
    def load_model(self,modelPath=None):
        #modelPath = "D:\\ai\\download\\chatglm3"
        #modelPath = "I:\wd0717\wendaMain\wenda\model\chatglm3-6b"
        #配置分词器
        tokenizer = AutoTokenizer.from_pretrained(modelPath,trust_remote_code=True,use_fast=True)

        #加载模型
        # model = AutoModel.from_pretrained(modelPath,trust_remote_code=True,device_map="auto")
        model = AutoModel.from_pretrained(modelPath,trust_remote_code=True,device_map="auto",offload_folder="/home/vincent/offload")  # 指定一个临时目录来存放卸载的权重
        # model = AutoModel.from_pretrained(modelPath,trust_remote_code=True,device_map="cpu")
        model = model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        
    def _call(self,prompt,config={},history=[]):
        return self.invoke(prompt,history)
    
    def invoke(self,prompt,config={},history=[]):
        if not isinstance(prompt, str):
            prompt = prompt.to_string()
        response,history = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature
        )
        self.history = history
        return AIMessage(content=response)
        
    def stream(self,prompt,config={},history=[]):
        if not isinstance(prompt, str):
            prompt = prompt.to_string()
        preResponse = ""
        for response,new_history in self.model.stream_chat(self.tokenizer,prompt):
            #self.history = new_history
            
            if preResponse == "":
                result = response
            else:
                result = response[len(preResponse):]
            preResponse = response
            yield result
        

llm = ChatGLM3()
# modelPath = "I:\wd0717\wendaMain\wenda\model\chatglm3-6b"

# modelPath = "/home/vincent/.lmstudio/models/chatglm3-6b-base"
modelPath = "/home/vincent/.lmstudio/models/chatglm3-6b-base"
# modelPath = "/home/vincent/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF"

llm.load_model(modelPath)

#调用call方法
llm.invoke("中国的首都是？")

for response in llm.stream("写一首诗春节的诗"):
    print(response,end="")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("请根据下面的主题写一篇小红书营销的短文： {topic}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# chain.invoke({"topic": "康师傅绿茶"})

for chunk in chain.stream({"topic": "康师傅绿茶"}):
    print(chunk,end="")





# 
# 
# 
# 
# 




