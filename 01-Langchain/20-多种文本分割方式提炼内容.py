from langchain_community.document_loaders import TextLoader
import os

# 1. 分割txt
loader = TextLoader("./02-agent/txt/faq-4359.txt",encoding="utf8")
doc = loader.load()
print(doc)

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([doc[0].page_content])
print(texts)

# 2. 分割编程语言
# ## Split code 拆分代码
# 
# CodeTextSplitter 允许您使用支持的多种语言拆分代码。导入枚举 Language 并指定语言。

from langchain_text_splitters import (Language, RecursiveCharacterTextSplitter)

# Full list of supported languages
[e.value for e in Language]
print([e.value for e in Language])

RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)

loader = TextLoader("./01-Langchain/js/main.js",encoding="utf8")
doc = loader.load()
print(doc)

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=250, chunk_overlap=20
)
js_docs = js_splitter.create_documents([doc[0].page_content])
print(js_docs)


# 3. 分割Markdown
# ## Markdown文本分割器
# 
# 许多聊天或问答应用程序都涉及在嵌入和矢量存储之前对输入文档进行分块。
# 
# 如前所述，分块通常旨在将具有共同上下文的文本放在一起。考虑到这一点，我们可能希望特别尊重文档本身的结构。例如，Markdown 文件是按标题组织的。在特定标头组中创建块是一个直观的想法。为了解决这个挑战，我们可以使用 MarkdownHeaderTextSplitter 。这将按一组指定的标头拆分 Markdown 文件。
# 
# 例如，如果我们想分割这个 markdown：
# 
# ```
# # Foo
# ## Bar
# Hi this is Jim  
# 
# Hi this is Joe
# 
# ## Baz
# Hi this is Molly
# ```
# 

from langchain_text_splitters import MarkdownHeaderTextSplitter

# loader = TextLoader("../02-agent/txt/stable_diffusion.md",encoding="utf8")
loader = TextLoader("./02-agent/txt/yjhx.md", encoding="utf8")
doc = loader.load()
print(doc)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(doc[0].page_content)
print(md_header_splits)


markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)
md_header_splits = markdown_splitter.split_text(doc[0].page_content)
print(md_header_splits)




