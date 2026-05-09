"""
项目评估脚本 v4 - 3 路对比 + LLM-as-Judge
功能：
1. 从 evaluation/test_questions.json 读取测试集（60 题）
2. 三种配置对比：
   - no_rag: 纯 LLM，不使用知识库
   - basic_rag: 简单 RAG，top_k=1
   - optimized_rag: 优化 RAG，top_k=3 + 改进 Prompt + CoT
3. LLM-as-Judge 评分
"""

import os
import json
import time
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv(override=True)

# ============ 加载测试集 ============

TEST_SET_PATH = "evaluation/test_questions.json"
with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
    test_questions = json.load(f)

print(f"📚 已加载测试集：{len(test_questions)} 道题")


# ============ 共享组件 ============

qa_llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

judge_llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

embedding = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
vectorstore = Chroma(persist_directory="./db", embedding_function=embedding)


# ============ Judge ============

REFUSAL_SIGNALS = ["知识库", "没有找到", "未找到", "不存在", "无法提供",
                   "不在考纲", "无相关", "未在", "拒绝", "没有相关",
                   "超出范围", "不在范围", "无法回答"]


def is_refusal(answer: str) -> bool:
    return any(sig in answer for sig in REFUSAL_SIGNALS)


def llm_judge_score(question, expected_answer, key_points, model_answer):
    kp_text = "\n".join(f"- {kp}" for kp in key_points) if key_points else "（无具体要点）"
    judge_prompt = f"""你是一个严格但公正的计算机考研408阅卷老师。请判断学生的回答是否基本正确。

【问题】
{question}

【参考答案】
{expected_answer}

【核心得分要点】
{kp_text}

【学生回答】
{model_answer}

【评分标准】
- 学生回答覆盖了核心要点的主要部分（≥60%），且没有重大事实错误 → PASS
- 学生回答严重遗漏关键要点，或包含明显错误 → FAIL

请只输出一个词：PASS 或 FAIL。不要解释。"""
    try:
        response = judge_llm.invoke(judge_prompt).content.strip().upper()
        return "PASS" in response
    except Exception as e:
        print(f"     ⚠️  Judge 出错：{str(e)[:40]}")
        return False


def evaluate_answer(question, answer, item):
    cat = item["category"]
    if cat == "hallucination":
        refused = is_refusal(answer)
        return refused, ("拒答正确" if refused else "未拒答")
    is_pass = llm_judge_score(
        question=question,
        expected_answer=item.get("expected_answer", ""),
        key_points=item.get("key_points", []),
        model_answer=answer
    )
    return is_pass, ("LLM裁判通过" if is_pass else "LLM裁判未通过")


# ============ 三种配置 ============

# 配置 1：纯 LLM
NO_RAG_PROMPT = """你是一个计算机考研408辅导助手。请回答以下问题。如果问题超出408考研范围，请明确说明"该问题不在408考研范围内"。

问题：{question}

回答："""

def answer_no_rag(question):
    return qa_llm.invoke(NO_RAG_PROMPT.format(question=question)).content


# 配置 2：简单 RAG
BASIC_RAG_PROMPT = """{context}

{question}"""

basic_rag_chain = RetrievalQA.from_chain_type(
    llm=qa_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=BASIC_RAG_PROMPT,
            input_variables=["context", "question"]
        )
    }
)

def answer_basic_rag(question):
    return basic_rag_chain.invoke({"query": question})["result"]


# 配置 3：优化 RAG（CoT + 不过度拒答）
OPTIMIZED_RAG_PROMPT = """你是一个专业的计算机考研408辅导助手。

参考资料：
{context}

问题：{question}

回答要求：
1. 优先基于参考资料回答；参考资料不足时，结合你的专业知识补充
2. 仅当问题明显超出408考研范围（如政治、生物、娱乐等）时，回答"该问题不在408考研范围内"
3. 涉及计算的题目，请展示推导步骤（思考过程）
4. 多要点问题请用编号列出

回答："""

optimized_rag_chain = RetrievalQA.from_chain_type(
    llm=qa_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=OPTIMIZED_RAG_PROMPT,
            input_variables=["context", "question"]
        )
    }
)

def answer_optimized_rag(question):
    return optimized_rag_chain.invoke({"query": question})["result"]


# ============ 评估主循环 ============

def run_evaluation(config_name, answer_fn):
    print(f"\n{'=' * 60}")
    print(f"🔬 {config_name}")
    print(f"{'=' * 60}\n")

    results = []
    for i, item in enumerate(test_questions):
        q = item["question"]
        subj, cat = item["subject"], item["category"]
        print(f"[{i+1:>2}/{len(test_questions)}] [{subj}/{cat}] {q[:35]}...")
        try:
            answer = answer_fn(q)
            is_pass, reason = evaluate_answer(q, answer, item)
            print(f"     {'✅' if is_pass else '❌'}  {reason}")
            results.append({
                "id": item["id"], "subject": subj, "category": cat,
                "question": q, "answer": answer,
                "expected_answer": item.get("expected_answer", ""),
                "key_points": item.get("key_points", []),
                "is_pass": is_pass, "judge_reason": reason
            })
        except Exception as e:
            print(f"     ❌  执行出错：{str(e)[:50]}")
            results.append({
                "id": item["id"], "subject": subj, "category": cat,
                "question": q, "answer": f"[ERROR] {str(e)}",
                "expected_answer": item.get("expected_answer", ""),
                "key_points": item.get("key_points", []),
                "is_pass": False, "judge_reason": "执行出错"
            })
        time.sleep(0.3)

    total_n = len(results)
    pass_n = sum(1 for r in results if r["is_pass"])
    accuracy = pass_n / total_n * 100

    by_subject = defaultdict(lambda: [0, 0])
    by_category = defaultdict(lambda: [0, 0])
    for r in results:
        by_subject[r["subject"]][1] += 1
        by_category[r["category"]][1] += 1
        if r["is_pass"]:
            by_subject[r["subject"]][0] += 1
            by_category[r["category"]][0] += 1

    print(f"\n--- {config_name} 总览 ---")
    print(f"📊 总准确率：{accuracy:.1f}% ({pass_n}/{total_n})\n")
    print("📂 按科目：")
    for k, (p, t) in by_subject.items():
        print(f"   {k:<22} {p:>2}/{t:<2}  ({p/t*100:.1f}%)")
    print("\n🏷️  按题型：")
    for k, (p, t) in by_category.items():
        print(f"   {k:<15} {p:>2}/{t:<2}  ({p/t*100:.1f}%)")

    return {
        "config": config_name,
        "accuracy": round(accuracy, 2),
        "pass_count": pass_n, "total": total_n,
        "by_subject": {k: {"pass": p, "total": t, "accuracy": round(p/t*100, 2)}
                       for k, (p, t) in by_subject.items()},
        "by_category": {k: {"pass": p, "total": t, "accuracy": round(p/t*100, 2)}
                        for k, (p, t) in by_category.items()},
        "details": results
    }


# ============ 主流程 ============

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("RAG 评估开始（60 题 × 3 配置 · LLM-as-Judge）")
    print("🚀" * 30)

    no_rag = run_evaluation("Config 1: no_rag (纯 LLM)", answer_no_rag)
    basic_rag = run_evaluation("Config 2: basic_rag (简单 RAG, top_k=1)", answer_basic_rag)
    optimized_rag = run_evaluation("Config 3: optimized_rag (CoT + top_k=3)", answer_optimized_rag)

    print("\n" + "=" * 60)
    print("📊 三路评估对比")
    print("=" * 60)
    print(f"  Config 1 (纯 LLM)        :  {no_rag['accuracy']:>5.1f}% ({no_rag['pass_count']:>2}/{no_rag['total']})")
    print(f"  Config 2 (简单 RAG)      :  {basic_rag['accuracy']:>5.1f}% ({basic_rag['pass_count']:>2}/{basic_rag['total']})")
    print(f"  Config 3 (优化 RAG)      :  {optimized_rag['accuracy']:>5.1f}% ({optimized_rag['pass_count']:>2}/{optimized_rag['total']})")
    print(f"\n🎯 提升路径：")
    print(f"   纯 LLM → 加 RAG       : {basic_rag['accuracy'] - no_rag['accuracy']:+.1f}%")
    print(f"   加 RAG → 优化 RAG     : {optimized_rag['accuracy'] - basic_rag['accuracy']:+.1f}%")
    print(f"   总提升                 : {optimized_rag['accuracy'] - no_rag['accuracy']:+.1f}%")

    summary = {
        "test_set_size": len(test_questions),
        "evaluation_method": "LLM-as-Judge (qwen-plus)",
        "configs": {
            "no_rag": {k: v for k, v in no_rag.items() if k != "details"},
            "basic_rag": {k: v for k, v in basic_rag.items() if k != "details"},
            "optimized_rag": {k: v for k, v in optimized_rag.items() if k != "details"}
        },
        "improvements": {
            "rag_vs_no_rag": round(basic_rag["accuracy"] - no_rag["accuracy"], 2),
            "optimized_vs_basic": round(optimized_rag["accuracy"] - basic_rag["accuracy"], 2),
            "total": round(optimized_rag["accuracy"] - no_rag["accuracy"], 2)
        }
    }
    with open("evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open("evaluation/evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump({
            "no_rag_details": no_rag["details"],
            "basic_rag_details": basic_rag["details"],
            "optimized_rag_details": optimized_rag["details"]
        }, f, ensure_ascii=False, indent=2)

    print("\n💾 结果已保存")
