# 导入工具
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从.env文件加载API Key（override=True 强制覆盖系统环境变量）
load_dotenv(override=True)

# 创建大模型对象，连接阿里云百炼
llm = ChatOpenAI(
    model="qwen-plus",                                                    # 用通义千问的plus模型
    api_key=os.getenv("DASHSCOPE_API_KEY"),                               # 从.env读取Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",         # 百炼的服务地址
    temperature=0
)

# 发送一个问题
response = llm.invoke("请用一句话介绍你自己")

# 打印回答
print(response.content)
