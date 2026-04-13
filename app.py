from pathlib import Path
import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="DLOU Freshman Agent MVP")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def find_excel_file() -> Path:
    xlsx_files = list(DATA_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError("data 文件夹下没有找到 .xlsx 文件")
    return xlsx_files[0]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_multi_values(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[，,；;/、\n\r]+", value)
    return [p.strip() for p in parts if p.strip()]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_knowledge_base() -> list[dict]:
    excel_file = find_excel_file()
    sheets = pd.read_excel(excel_file, sheet_name=None)
    kb_df = None

    # 优先找 knowledge_base 工作表
    for sheet_name, df in sheets.items():
        df = normalize_columns(df)
        cols = set(df.columns)
        if sheet_name.lower() == "knowledge_base":
            kb_df = df
            break

    # 找不到的话，就找包含关键列的表
    if kb_df is None:
        for _, df in sheets.items():
            df = normalize_columns(df)
            cols = set(df.columns)
            if {"标准问题", "标准答案"}.issubset(cols):
                kb_df = df
                break

    if kb_df is None:
        raise ValueError("Excel 中未找到 knowledge_base 工作表，或缺少“标准问题/标准答案”列")

    kb_df = kb_df.fillna("")

    records = []
    for _, row in kb_df.iterrows():
        item = {str(k).strip(): normalize_text(v) for k, v in row.to_dict().items()}

        aliases_raw = item.get("问题别名/同义问法", "")
        aliases = split_multi_values(aliases_raw)

        campuses = split_multi_values(item.get("适用校区", ""))
        stages = split_multi_values(item.get("适用阶段", ""))

        item["_aliases"] = aliases
        item["_campuses"] = campuses
        item["_stages"] = stages
        records.append(item)

    return records


KB = load_knowledge_base()
MATCH_THRESHOLD = 5.0


def extract_keywords(question: str) -> list[str]:
    q = re.sub(r"\s+", "", question)
    raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", q)

    keywords = []
    stopwords = {"怎么", "如何", "请问", "一下", "一下子", "可以", "吗", "呢", "的", "我", "想", "知道"}
    for token in raw_tokens:
        token = token.strip()
        if not token or token in stopwords:
            continue
        # 尝试把长句拆成更有用的小片段
        sub_parts = re.split(r"(怎么|如何|请问|可以|吗|呢)", token)
        for part in sub_parts:
            part = part.strip()
            if len(part) >= 2 and part not in stopwords:
                keywords.append(part)

    # 去重保序
    seen = set()
    result = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def match_campus(item: dict, campus: str) -> bool:
    campus = campus.strip()
    if not campus:
        return True
    item_campuses = item.get("_campuses", [])
    if not item_campuses:
        return True
    if "全校通用" in item_campuses:
        return True
    return campus in item_campuses


def match_stage(item: dict, stage: str) -> bool:
    stage = stage.strip()
    if not stage:
        return True
    item_stages = item.get("_stages", [])
    if not item_stages:
        return True
    return stage in item_stages


def score_item(question: str, item: dict) -> float:
    question = question.strip()
    std_q = item.get("标准问题", "")
    aliases = item.get("_aliases", [])
    cat1 = item.get("一级分类", "")
    cat2 = item.get("二级分类", "")
    answer = item.get("标准答案", "")

    score = 0.0

    # 1. 直接包含关系
    if question and question in std_q:
        score += 8
    if question and any(question in alias for alias in aliases):
        score += 7
    if question and question in answer:
        score += 4

    # 2. 相似度
    score += text_similarity(question, std_q) * 6
    for alias in aliases:
        score += text_similarity(question, alias) * 4

    # 3. 关键词命中
    keywords = extract_keywords(question)
    texts = [std_q, cat1, cat2, answer] + aliases
    for kw in keywords:
        for text in texts:
            if kw and kw in text:
                score += 2

    # 4. 分类命中加分
    if cat1 and cat1 in question:
        score += 2
    if cat2 and cat2 in question:
        score += 2

    return score


def search_knowledge(question: str, campus: str = "", stage: str = "") -> tuple[dict | None, float]:
    candidates = [item for item in KB if match_campus(item, campus) and match_stage(item, stage)]

    # 如果过滤后没有候选，就退回全量知识库
    if not candidates:
        candidates = KB

    best_item = None
    best_score = -1.0

    for item in candidates:
        s = score_item(question, item)
        if s > best_score:
            best_score = s
            best_item = item

    return best_item, best_score


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    campuses = ["", "黑石礁", "大黑石", "瓦房店"]
    stages = ["", "入学前", "报到当天", "第一周", "第一个月"]
    context = {
        "request": request,
        "campuses": campuses,
        "stages": stages,
        "kb_count": len(KB),
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
    )


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = str(data.get("question", "")).strip()
    campus = str(data.get("campus", "")).strip()
    stage = str(data.get("stage", "")).strip()

    if not question:
        return JSONResponse(
            {
                "matched": False,
                "message": "请输入问题。",
            },
            status_code=400,
        )

    # item, score = search_knowledge(question, campus, stage)

    # if not item or score < 3:
    #     return {
    #         "matched": False,
    #         "message": "暂未找到完全匹配的校本知识，请换一种问法，或以学校最新通知为准。",
    #         "score": round(score, 2),
    #     }
    item, score = search_knowledge(question, campus, stage)

    if not item or score < MATCH_THRESHOLD:
        return {
            "matched": False,
            "message": "暂未找到完全匹配的校本知识，请换一种问法，或以学校最新通知为准。",
            "score": round(score, 2),
        }



    return {
        "matched": True,
        "score": round(score, 2),
        "item_id": item.get("id", ""),
        "standard_question": item.get("标准问题", ""),
        "category_1": item.get("一级分类", ""),
        "category_2": item.get("二级分类", ""),
        "answer": item.get("标准答案", ""),
        "steps": item.get("办理步骤", ""),
        "campus": item.get("适用校区", ""),
        "stage": item.get("适用阶段", ""),
        "official_page": item.get("官方入口/页面", ""),
        "contact": item.get("联系方式", ""),
        "materials": item.get("所需材料", ""),
        "notice": item.get("注意事项", ""),
        "need_manual_confirm": item.get("是否需要人工确认", ""),
        "source_title": item.get("来源标题", ""),
        "source_link": item.get("来源链接", ""),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "kb_count": len(KB)}