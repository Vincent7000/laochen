# %% [markdown]
# ## DuckDuckGo 
# DuckDuckGo 是一款注重隐私的网络搜索引擎，由 Gabriel Weinberg 创立于 2008 年。这款搜索引擎的特点在于不存储用户个人信息，不跟踪用户的搜索历史，从而保护用户的隐私。
# 
# 以下是 DuckDuckGo 的一些主要特点：
# 
# 1. 隐私保护：DuckDuckGo 不存储用户搜索历史，不跟踪用户，不建立用户档案。这意味着用户在搜索时可以保持匿名，不用担心搜索历史被用于广告或其他目的。
# 2. 清晰的搜索结果：DuckDuckGo 的搜索结果页设计简洁，易于阅读。它还提供了一些独特的功能，如“零点击信息”（Zero-click Info），可以直接在搜索结果页显示相关信息，无需点击进入其他网页。
# 3. 集成其他服务：DuckDuckGo 集成了许多其他服务，如gMaps、gMaps Street View、Wikipedia、YouTube等。用户可以直接在搜索结果页使用这些服务，无需打开其他网站。
# 4. 搜索速度快：DuckDuckGo 的搜索速度较快，能够快速响应用户的搜索请求。
# 5. 移动应用：DuckDuckGo 提供了移动应用，用户可以在智能手机或平板电脑上使用这款搜索引擎。
# 6. 浏览器扩展：DuckDuckGo 还提供了一些浏览器扩展，如DuckDuckGo Privacy Essentials，可以帮助用户更好地保护隐私。
# 
# 然而，DuckDuckGo 在中国大陆无法直接访问。如果您在中国大陆地区，建议使用合规的搜索引擎，如百度、搜狗等。
# 

# %%
from langchain_community.tools import DuckDuckGoSearchRun

# %%
search = DuckDuckGoSearchRun()

# %%
search.run("蔡徐坤")

# %% [markdown]
# 要获取更多附加信息（例如链接、来源），请使用 DuckDuckGoSearchResults()

# %%
from langchain_community.tools import DuckDuckGoSearchResults

# %%
search = DuckDuckGoSearchResults()

# %%
search.run("刘亦菲")

# %% [markdown]
# 您也可以只搜索新闻文章。使用关键字 backend="news"

# %%
search = DuckDuckGoSearchResults(backend="news")

# %%
search.run("刘亦菲")

# %% [markdown]
# 您也可以直接将自定义 DuckDuckGoSearchAPIWrapper 传递给 DuckDuckGoSearchResults 。因此，您可以更好地控制搜索结果。

# %%
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", max_results=2,source="news")

# %%
search = DuckDuckGoSearchResults(api_wrapper=wrapper, )
search.run("刘亦菲")

# %% [markdown]
# https://pypi.org/project/duckduckgo-search/
# ```
# param max_results: int = 5 
# param region: Optional[str] = 'wt-wt'
# param source: str = 'text'
# ```       

# %%



