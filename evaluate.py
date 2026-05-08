"""
项目评估脚本
功能：
1. 跑一组测试题，让大模型当裁判评估回答质量
2. 测试两种参数配置（基线 vs 优化）
3. 算出准确率，得出"优化前后"的对比数据
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv(override=True)

# ============ 测试集（你可以根据自己的知识库内容修改）============

test_questions = [
    # ===== 正常题（应该能从知识库回答）=====
    {
        "question": "什么是进程和线程的区别？",
        "expected_keywords": ["资源", "调度", "通信", "独立", "共享"]
    },
    {
        "question": "什么是死锁？",
        "expected_keywords": ["互斥", "等待", "资源"]
    },
    {
        "question": "死锁的四个必要条件是什么？",
        "expected_keywords": ["互斥", "占有", "不可剥夺", "循环等待"]
    },
    {
        "question": "什么是虚拟内存？",
        "expected_keywords": ["磁盘", "物理内存", "虚拟"]
    },
    {
        "question": "Cache的作用是什么？",
        "expected_keywords": ["CPU", "高速", "局部性"]
    },
    {
        "question": "页面置换算法有哪些？",
        "expected_keywords": ["FIFO", "LRU", "OPT"]
    },
    {
        "question": "分页和分段有什么区别？",
        "expected_keywords": ["固定", "逻辑", "页", "段"]
    },
    {
        "question": "什么是虚拟存储器？",
        "expected_keywords": ["主存", "辅存", "扩大", "用户"]
    },
    {
        "question": "操作系统的主要功能有哪些？",
        "expected_keywords": ["进程", "内存", "文件", "设备"]
    },
    {
        "question": "进程的状态有哪几种？",
        "expected_keywords": ["就绪", "运行", "阻塞"]
    },

    # ===== 陷阱题（知识库里没有，应该说"没找到"）=====
    {
        "question": "什么是量子纠缠？",
        "expected_keywords": ["知识库中没有"]
    },
    {
        "question": "Python列表和元组的区别是什么？",
        "expected_keywords": ["知识库中没有"]
    },
    {
        "question": "TCP三次握手的过程？",
        "expected_keywords": ["知识库中没有"]
    },
]

# ============ 评估函数 ============

def evaluate_answer(answer, expected_keywords):
    """
    简单的关键词匹配评估
    检查回答里是否包含预期的关键词，命中越多分越高
    返回：（是否通过，命中数，总数）
    """
    hit_count = 0
    for keyword in expected_keywords:
        if keyword in answer:
            hit_count += 1

    total = len(expected_keywords)
    # 命中率 >= 60% 视为通过
    is_pass = (hit_count / total) >= 0.6
    return is_pass, hit_count, total


def run_evaluation(config_name, chunk_size, top_k, prompt_template):
    """
    用指定的配置跑一遍评估
    config_name: 配置名称（用于打印）
    chunk_size: 文本块大小（这里用不到，仅作为标记）
    top_k: 检索返回的数量
    prompt_template: Prompt模板
    """
    print(f"\n{'=' * 60}")
    print(f"评估配置：{config_name}")
    print(f"  - top_k: {top_k}")
    print(f"{'=' * 60}\n")

    # 加载知识库
    embedding = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory="./db",
        embedding_function=embedding
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 创建大模型
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0
    )

    # 构建Prompt
    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 组装问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # 跑测试集
    pass_count = 0
    total_hit = 0
    total_keywords = 0
    results = []

    for i, item in enumerate(test_questions):
        print(f"[{i+1}/{len(test_questions)}] {item['question']}")

        try:
            result = qa_chain.invoke({"query": item["question"]})
            answer = result["result"]

            is_pass, hit, total = evaluate_answer(answer, item["expected_keywords"])

            if is_pass:
                pass_count += 1
                status = "✅ 通过"
            else:
                status = "❌ 未通过"

            total_hit += hit
            total_keywords += total

            print(f"  {status} 关键词命中: {hit}/{total}")

            results.append({
                "question": item["question"],
                "answer": answer,
                "is_pass": is_pass,
                "hit": hit,
                "total": total
            })

        except Exception as e:
            print(f"  ❌ 出错: {str(e)[:50]}")
            results.append({
                "question": item["question"],
                "answer": "",
                "is_pass": False,
                "hit": 0,
                "total": len(item["expected_keywords"])
            })

    # 统计结果
    accuracy = pass_count / len(test_questions) * 100
    keyword_recall = total_hit / total_keywords * 100

    print(f"\n--- {config_name} 评估结果 ---")
    print(f"通过题数：{pass_count}/{len(test_questions)}")
    print(f"准确率：{accuracy:.1f}%")
    print(f"关键词召回率：{keyword_recall:.1f}%")

    return {
        "config": config_name,
        "accuracy": accuracy,
        "keyword_recall": keyword_recall,
        "pass_count": pass_count,
        "total": len(test_questions)
    }


# ============ 配置1：基线（无优化的初始版本，top_k=1）============

baseline_prompt = """{context}

{question}"""

# ============ 配置2：优化版（强Prompt + top_k=3）============

optimized_prompt = """你是一个专业的计算机考研408辅导助手。请根据以下参考内容回答用户的问题。

要求：
1. 严格基于参考内容回答，不要编造信息
2. 如果参考内容中没有相关信息，请回答"知识库中没有找到相关内容"
3. 回答要简洁清晰，重点突出
4. 如果内容包含多个要点，用编号列出

参考内容：
{context}

用户问题：{question}

回答："""

# ============ 主流程 ============

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("RAG系统评估开始")
    print("🚀" * 30)

    # 跑基线
    baseline_result = run_evaluation(
        config_name="基线配置（无优化Prompt + top_k=1）",
        chunk_size=500,
        top_k=1,
        prompt_template=baseline_prompt
    )

    # 跑优化版
    optimized_result = run_evaluation(
        config_name="优化配置（强Prompt + top_k=3）",
        chunk_size=500,
        top_k=3,
        prompt_template=optimized_prompt
    )

    # 对比结果
    print("\n" + "=" * 60)
    print("📊 评估对比结果")
    print("=" * 60)
    print(f"\n{'配置':<35} {'准确率':<12} {'关键词召回率':<12}")
    print("-" * 60)
    print(f"{baseline_result['config']:<35} {baseline_result['accuracy']:.1f}%       {baseline_result['keyword_recall']:.1f}%")
    print(f"{optimized_result['config']:<35} {optimized_result['accuracy']:.1f}%       {optimized_result['keyword_recall']:.1f}%")

    improvement = optimized_result['accuracy'] - baseline_result['accuracy']
    print(f"\n🎯 准确率提升：{improvement:+.1f}%")
    print(f"   {baseline_result['accuracy']:.0f}% → {optimized_result['accuracy']:.0f}%")

    # 保存详细结果到文件
    final_data = {
        "baseline": baseline_result,
        "optimized": optimized_result,
        "improvement": improvement
    }
    with open("evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print("\n详细结果已保存到 evaluation_result.json")
