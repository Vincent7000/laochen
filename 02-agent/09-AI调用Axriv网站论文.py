# %% [markdown]
# ## arXiv
# 
# arXiv是一个由康奈尔大学维护的在线预印本论文存储库，它提供了物理、数学、计算机科学、定量生物学、定量金融学和统计学的开放获取服务。arXiv成立于1991年，最初是由物理学家保罗·金斯帕格（Paul Ginsparg）创建，目的是为了提供一个电子化的方式供物理学家共享研究成果。随着时间的推移，它逐渐扩展到其他学科。
# ### arXiv的特点和功能
# 1. **预印本平台**：arXiv允许科研人员在将论文提交给学术期刊发表之前，先行发布其研究成果。这样做的目的是为了加快科学知识的传播。
# 2. **开放获取**：所有在arXiv上发表的论文都可以免费获取，这有助于全球的研究人员、学者和学生获取最新的科学进展。
# 3. **同行评审**：虽然arXiv上的论文未经正式的同行评审，但它们通常会在提交前由arXiv的志愿者或编委会进行审核，以确保论文的基本质量和主题相关性。
# 4. **分类和标签**：论文按照学科分类，并且可以使用关键词进行检索，便于用户找到自己感兴趣的研究领域。
# 5. **版本控制**：作者可以上传论文的新版本，更新研究成果或回应同行评审的反馈。每个版本都会被记录，确保了研究过程的透明性。
# 6. **引用和统计**：arXiv提供论文的引用次数和下载次数，这可以作为衡量论文影响力的一个指标。
# ### 如何使用arXiv
# 1. **浏览和搜索**：用户可以直接在arXiv网站上浏览最新上传的论文，或者使用搜索功能查找特定主题的论文。
# 2. **订阅和通知**：用户可以订阅特定主题或作者的更新，当有新的论文上传时，arXiv会通过电子邮件通知订阅者。
# 3. **提交论文**：研究人员可以通过arXiv网站提交自己的论文。提交前需要注册账号，并遵守arXiv的提交指南。
# 4. **评论和讨论**：虽然arXiv本身不提供评论功能，但有些第三方平台允许用户对arXiv上的论文进行评论和讨论。
# ### arXiv在中国的影响
# 在中国，arXiv同样被广泛使用，它为国内外的科研人员提供了一个宝贵的信息共享平台。中国的科研机构和大学鼓励研究人员使用arXiv来展示他们的研究成果，以促进国际合作和学术交流。同时，中国的科研人员也通过arXiv获取全球科学研究的最新进展。
# arXiv对全球科学研究的开放性和可获取性作出了重要贡献，符合全球科学共同体共同推动科学知识传播和学术交流的愿景。
# 

# %%
from langchain_openai import ChatOpenAI, OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1234/v1"
model = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    temperature=0.1,
)
llm = model

# %%
from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivAPIWrapper()
docs = arxiv.run("1605.08386")
docs

# %%
from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivAPIWrapper()
docs = arxiv.run("sora")
docs

# %%
import arxiv

search = arxiv.Search(
    query = "gpt4",
    max_results = 5,
    sort_by = arxiv.SortCriterion.Relevance
)
search

# %%
client = arxiv.Client()
results = client.results(search)

results

# %%
papers = []
for item in results:
    print(item)
    papers.append(item)

# %%
papers[0]

# %%
htmlUrls = []

for item in papers:
    url = item.entry_id.replace("abs","html")
    htmlUrls.append(url)

    
htmlUrls


# %%
import urllib.parse

url = "http://arxiv.org/html/2309.12732v1"
# 使用urllib.parse.urlsplit来处理URL，它会更智能地处理带参数的URL
url_parts = urllib.parse.urlsplit(url)

# 获取路径部分
path = url_parts.path

# 分割路径来获取最后一部分
filename = path.split('/')[-1]

print(filename)  # 输出



# %%
from langchain_community.document_loaders import ArxivLoader
docs = ArxivLoader(query="2309.12732v1", load_max_docs=2).load()
docs

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("{article}\n\n\n请使用中文详细讲解上面这篇文章内容,并将核心的要点提炼出来")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# %%
chain.invoke({"article":docs[0].page_content})

# %%



