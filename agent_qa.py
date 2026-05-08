"""
Agent问答系统
功能：
1. Agent自主决策——根据问题选择"知识库检索"或"网络搜索"工具
2. 多轮对话记忆——记住之前聊过的内容，支持代词消解
"""

import os
from dotenv import load_dotenv

# 大模型与Embedding
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# Agent相关
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 网络搜索工具
from langchain_tavily import TavilySearch

# 多轮对话记忆
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 加载.env里的API Key
load_dotenv(override=True)

# ============ 第一步：加载知识库（和RAG那个一样）============

print("正在加载知识库...")

embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("知识库加载完成")

# ============ 第二步：定义两个工具 ============

# 工具1：知识库检索
# @tool 装饰器把一个普通函数变成Agent能用的"工具"
# 函数的注释（"""..."""）非常重要，Agent靠它判断什么时候该用这个工具
@tool
def search_knowledge_base(query: str) -> str:
    """
    在计算机考研408知识库中检索内容。
    适用场景：当用户的问题涉及计算机基础知识、408考研内容
    （操作系统、数据结构、计算机网络、计算机组成原理）时使用此工具。
    参数 query: 用户的问题或检索关键词
    """
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "知识库中没有找到相关内容"
    # 把检索到的所有文档拼成一段文本返回
    result = "\n\n".join([f"【片段{i+1}】{doc.page_content}" for i, doc in enumerate(docs)])
    return result

# 工具2：网络搜索
# 直接用Tavily提供的搜索工具
web_search = TavilySearch(
    max_results=3,                # 返回3条搜索结果
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

# 修改一下网络搜索工具的描述，让Agent知道什么时候该用它
web_search.description = (
    "在网络上搜索实时信息。"
    "适用场景：当用户问的是知识库中没有的内容时使用，"
    "比如实时新闻、当前事件、最新的技术动态、天气、股价等。"
    "参数 query: 搜索关键词"
)

# 把两个工具放进列表
tools = [search_knowledge_base, web_search]

# ============ 第三步：创建大模型 ============

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

# ============ 第四步：创建Agent的Prompt模板 ============

# Agent的Prompt和普通RAG不一样，要包含几个特殊位置：
# {chat_history}：对话历史（多轮对话记忆用）
# {input}：用户当前的输入
# {agent_scratchpad}：Agent的"草稿纸"——它思考和调用工具的中间过程
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的计算机考研408智能助手。

你有两个工具可以使用：
1. search_knowledge_base - 计算机考研408知识库（首选，回答考研相关问题）
2. tavily_search - 网络搜索（用于实时信息或知识库没有的内容）

回答规则：
1. 优先使用知识库工具回答考研相关问题
2. 知识库没有相关内容时，使用网络搜索
3. 始终基于工具返回的内容回答，不要编造
4. 回答要简洁清晰，重点突出
5. 涉及多个要点时用编号列出"""),
    MessagesPlaceholder(variable_name="chat_history"),    # 对话历史的占位符
    ("human", "{input}"),                                 # 用户输入
    MessagesPlaceholder(variable_name="agent_scratchpad"),# Agent思考过程的占位符
])

# ============ 第五步：创建Agent ============

# create_tool_calling_agent：创建一个支持工具调用的Agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor：负责真正运行Agent的执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,        # True=打印Agent的思考过程，False=只显示最终回答
    max_iterations=5     # 最多循环5次（防止无限循环）
)

# ============ 第六步：加上对话历史记忆 ============

# 用一个字典存所有会话的历史记录
# 一个用户ID对应一份对话历史
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取对话历史，没有就创建一个"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# 把Agent和对话历史包装在一起
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",        # 用户输入字段
    history_messages_key="chat_history" # 历史消息字段（要和prompt里的占位符一致）
)

# ============ 第七步：开始问答（循环）============

print("\n" + "=" * 50)
print("Agent问答系统已就绪，开始提问吧！")
print("特性：1.支持网络搜索  2.记住对话上下文")
print("输入 'quit' 或 'exit' 退出")
print("=" * 50 + "\n")

# 用一个固定的session_id（单用户场景）
session_id = "user_001"

while True:
    question = input("\n你的问题：").strip()

    if question.lower() in ["quit", "exit", "q"]:
        print("再见！")
        break

    if not question:
        continue

    print("\n思考中...")

    # 调用带历史的Agent
    result = agent_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

    print(f"\n【最终回答】")
    print(result["output"])
    print("\n" + "-" * 50)
