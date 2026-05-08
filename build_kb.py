"""
构建知识库 - 完整版
功能：加载文档 → 切分 → 向量化 → 存入数据库
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # 加载PDF和文本文件
from langchain_text_splitters import RecursiveCharacterTextSplitter        # 切分文本
from langchain_community.embeddings import DashScopeEmbeddings              # 阿里云的Embedding
from langchain_community.vectorstores import Chroma                          # 向量数据库

# 加载.env文件里的API Key
load_dotenv(override=True)

# ============ 第一步：加载所有文档 ============

data_folder = "data"
all_docs = []

# 遍历data文件夹里的所有文件
for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)

    # 处理 .pdf 文件
    if filename.endswith(".pdf"):
        print(f"正在加载PDF：{filename}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  → 加载了 {len(docs)} 页")

    # 处理 .md 和 .txt 文件
    elif filename.endswith(".md") or filename.endswith(".txt"):
        print(f"正在加载文本：{filename}")
        # encoding="utf-8" 指定编码，防止中文乱码
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  → 加载完成")

print(f"\n所有文档加载完毕，共 {len(all_docs)} 个文档")

# ============ 第二步：切分文本 ============

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_docs)
print(f"切分完成，共 {len(chunks)} 个文本块")

# ============ 第三步：向量化 + 存入数据库 ============

print("\n开始向量化（这一步会调用百炼的Embedding API，需要联网）...")

# 创建Embedding工具，用阿里云百炼的text-embedding-v2模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 把所有文本块向量化，存入Chroma数据库
# persist_directory：数据库保存的文件夹（之前创建的db文件夹）
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./db"
)

print(f"\n✅ 知识库构建完成！")
print(f"   总文本块数：{len(chunks)}")
print(f"   数据库位置：./db")

# ============ 第四步：测试一下检索效果 ============

print("\n--- 测试检索效果 ---")
test_question = "什么是进程和线程的区别？"
print(f"测试问题：{test_question}\n")

# 检索最相关的3段
results = vectorstore.similarity_search(test_question, k=3)

for i, doc in enumerate(results):
    print(f"--- 第{i+1}段 ---")
    print(doc.page_content[:200])  # 只显示前200字
    print(f"来源：{doc.metadata.get('source', '未知')}")
    print()
