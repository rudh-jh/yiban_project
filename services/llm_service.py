import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL_NAME = os.getenv("OPENAI_MODEL", "openrouter/free")

SYSTEM_PROMPT = """
你是“大连海洋大学新生适应智能体”的回答整理助手。

你的任务不是判断是否命中知识，而是：
把系统已经检索到的校本知识条目，整理成自然、清晰、简洁的回答。

必须严格遵守：
1. 只能根据提供的知识条目回答，不要编造新信息。
2. 不要说“当前知识库未覆盖该问题”，除非提供的知识条目内容明显为空。
3. 如果已有知识条目能回答，就直接基于这些条目作答。
4. 优先参考匹配分数最高的知识条目。
5. 回答采用下面格式：

【简要回答】
【办理步骤】
【注意事项】
【来源说明】
"""

def build_kb_context(items: List[Dict]) -> str:
    parts = []
    for idx, item in enumerate(items, start=1):
        parts.append(
            f"""[知识条目{idx}]
知识ID: {item.get("id", "")}
标准问题: {item.get("标准问题", "")}
标准答案: {item.get("标准答案", "")}
办理步骤: {item.get("办理步骤", "")}
适用校区: {item.get("适用校区", "")}
适用阶段: {item.get("适用阶段", "")}
官方入口/页面: {item.get("官方入口/页面", "")}
联系方式: {item.get("联系方式", "")}
所需材料: {item.get("所需材料", "")}
注意事项: {item.get("注意事项", "")}
来源标题: {item.get("来源标题", "")}
来源链接: {item.get("来源链接", "")}
匹配分数: {item.get("_score", "")}
"""
        )
    return "\n\n".join(parts)

def generate_grounded_answer(question: str, campus: str, stage: str, items: List[Dict]) -> str:
    kb_context = build_kb_context(items)

    user_input = f"""
用户问题：{question}
用户选择校区：{campus or "未指定"}
用户选择阶段：{stage or "未指定"}

下面已经是系统检索出的相关校本知识条目。
请你直接根据这些知识条目整理回答，优先使用匹配分数最高的条目。

{kb_context}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()