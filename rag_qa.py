"""
RAG问答系统
功能：用户提问 → 从知识库检索相关内容 → 大模型生成回答
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 加载.env里的API Key
load_dotenv(override=True)

# ============ 第一步：加载已经建好的知识库 ============

print("正在加载知识库...")

# 创建Embedding工具（要和建库时用的一样）
embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 加载之前保存的Chroma数据库
vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embedding
)

# 把数据库变成"检索器"
# k=3 表示每次检索返回最相关的3段内容
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("知识库加载完成")

# ============ 第二步：创建大模型 ============

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

# ============ 第三步：写Prompt模板 ============

# {context}会被自动替换成检索到的内容
# {question}会被自动替换成用户的问题
template = """你是一个专业的计算机考研408辅导助手。请根据以下参考内容回答用户的问题。

要求：
1. 严格基于参考内容回答，不要编造信息
2. 如果参考内容中没有相关信息，请回答"知识库中没有找到相关内容"
3. 回答要简洁清晰，重点突出
4. 如果内容包含多个要点，用编号列出

参考内容：
{context}

用户问题：
{question}

回答："""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# ============ 第四步：组装问答链 ============

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,                                       # 用哪个大模型
    chain_type="stuff",                             # 处理方式：把检索内容直接拼起来发给大模型
    retriever=retriever,                            # 用哪个检索器
    chain_type_kwargs={"prompt": QA_PROMPT},        # 用自定义的Prompt模板
    return_source_documents=True                    # 同时返回参考了哪些文档
)

# ============ 第五步：开始问答（循环） ============

print("\n" + "=" * 50)
print("RAG问答系统已就绪，开始提问吧！")
print("输入 'quit' 或 'exit' 退出")
print("=" * 50 + "\n")

# 进入循环，让你可以一直提问
while True:
    # 获取用户输入
    question = input("\n你的问题：").strip()

    # 退出条件
    if question.lower() in ["quit", "exit", "q"]:
        print("再见！")
        break

    # 空输入跳过
    if not question:
        continue

    print("\n思考中...")

    # 调用问答链
    result = qa_chain.invoke({"query": question})

    # 打印回答
    print(f"\n【回答】")
    print(result["result"])

    # 打印参考来源
    print(f"\n【参考来源】")
    sources = set()  # 用集合去重
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "未知")
        sources.add(source)
    for src in sources:
        print(f"  · {src}")

    print("\n" + "-" * 50)
