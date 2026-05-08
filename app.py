"""
Streamlit Web 界面
功能：把Agent问答系统包装成一个网页应用
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv(override=True)

# ============ 网页基础设置 ============

# 设置网页标题、图标、布局
st.set_page_config(
    page_title="408考研智能助手",
    page_icon="🎓",
    layout="wide"
)

# 网页主标题
st.title("🎓 计算机考研408智能助手")
st.caption("基于RAG + Agent的智能问答系统")

# ============ 缓存：让初始化只跑一次 ============

# @st.cache_resource 装饰器：被它装饰的函数，结果会被缓存
# 第一次运行时执行函数，之后直接用缓存的结果，避免每次刷新都重新加载数据库
@st.cache_resource
def init_agent():
    """初始化整个Agent系统，只执行一次"""

    # 加载知识库
    embedding = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory="./db",
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 定义工具
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        在计算机考研408知识库中检索内容。
        适用场景：回答计算机基础知识、408考研内容相关问题。
        """
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "知识库中没有找到相关内容"
        result = "\n\n".join([f"【片段{i+1}】{doc.page_content}" for i, doc in enumerate(docs)])
        return result

    web_search = TavilySearch(
        max_results=3,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    web_search.description = "在网络上搜索实时信息（天气、新闻、最新动态等）"

    tools = [search_knowledge_base, web_search]

    # 创建大模型
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0
    )

    # Prompt模板
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
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 创建Agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,         # 网页界面不需要打印思考过程
        max_iterations=5
    )

    # 加上对话历史
    session_store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return agent_with_history

# 初始化（只执行一次）
with st.spinner("正在加载知识库和Agent..."):
    agent_with_history = init_agent()

# ============ 用 session_state 管理对话历史 ============

# st.session_state：Streamlit的会话状态，可以在多次刷新间保留数据
# 第一次进入页面时初始化messages列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============ 侧边栏：清空对话按钮 ============

with st.sidebar:
    st.header("操作")

    # 显示一些项目信息
    st.markdown("### 项目说明")
    st.markdown("""
    本系统包含两个工具：
    - 📚 **知识库检索**：408考研知识
    - 🌐 **网络搜索**：实时信息

    Agent会根据问题自动选择工具。
    """)

    st.markdown("### 示例问题")
    st.markdown("""
    - 什么是死锁？
    - 进程和线程有什么区别？
    - 它有几个必要条件？（多轮对话）
    - 今天北京的天气怎么样？
    """)

    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============ 显示历史消息 ============

# 遍历所有历史消息，按用户/助手区分显示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============ 用户输入框 ============

# st.chat_input 是聊天专用的输入框，会自动出现在网页底部
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用Agent获取回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = agent_with_history.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "streamlit_user"}}
                )
                response = result["output"]
            except Exception as e:
                response = f"出错了：{str(e)}"

            st.markdown(response)

    # 把助手回答加入历史
    st.session_state.messages.append({"role": "assistant", "content": response})
