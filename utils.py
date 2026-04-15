from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

llm = ChatOpenAI(
    model="doubao-seed-1-6-flash-250828",
    temperature=0.3,
    api_key="eae9aeef-c953-466b-8cdd-5c98ee331ccd",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

@tool
def searxng_search(query):
    """输入搜索内容，使用 LLM 返回对查询主题的简单介绍。"""

        # SEARXNG_URL = 'http://127.0.0.1:6688/search'
    # params = {}
    # # 设置搜索参数
    # params['q'] = query
    # params['format'] = 'json'  # 返回 JSON 格式的结果
    # params['engines'] = 'bing'
    # # 发送 GET 请求
    # response = requests.get(SEARXNG_URL,params)
    # #return response.text
    # # 检查响应状态码
    # if response.status_code == 200:
    #     res = response.json()
    #     # print(res)
    #     resList = []
    #     for item in res['results']:
    #         resList.append({
    #             "title":item['title'],
    #             "content":item['content'],
    #             "url":item['url']
    #         })
    #         if len(resList) >= 3:
    #             break
    #     return resList 
    # else:
    #     response.raise_for_status()



    search_prompt = f"请简要介绍一下：{query}，控制在200字以内。"
    response = llm.invoke(search_prompt)
    return [{
        "title": f"关于 {query} 的介绍",
        "content": response.content,
        "url": ""
    }]



def render_text_description(tools):
    """渲染工具描述为文本格式"""
    descriptions = []
    for tool in tools:
        if hasattr(tool, 'description') and hasattr(tool, 'name'):
            if hasattr(tool, 'args'):
                args_str = str(tool.args)
                descriptions.append(f"{tool.name}: {tool.description}, args: {args_str}")
            else:
                descriptions.append(f"{tool.name}: {tool.description}")
        elif hasattr(tool, 'func') and hasattr(tool, 'description'):
            descriptions.append(f"{tool.name}: {tool.description}")
    return "\n".join(descriptions)