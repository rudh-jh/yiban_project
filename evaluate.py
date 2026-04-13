from pathlib import Path
from typing import Any
import pandas as pd

# 直接复用 app.py 里的检索函数
from app import find_excel_file, search_knowledge, MATCH_THRESHOLD


def norm(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()

def norm_id(v: Any) -> str:
    """
    把 Excel 里读出来的 1 / 1.0 / '1' / ' 1 ' 都统一成 '1'
    """
    if pd.isna(v):
        return ""

    s = str(v).strip()

    if not s:
        return ""

    # 如果是像 16.0 这种，转成 16
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def find_sheet(sheets: dict, target_name: str, required_cols: list[str] | None = None):
    # 优先按表名找
    for name, df in sheets.items():
        if str(name).strip().lower() == target_name.lower():
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            return df

    # 找不到就按关键列找
    if required_cols:
        for _, df in sheets.items():
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            cols = set(df.columns)
            if all(col in cols for col in required_cols):
                return df

    return None


def get_col(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row.index:
            return norm(row[name])
    return ""


def main():
    excel_file = find_excel_file()
    sheets = pd.read_excel(excel_file, sheet_name=None)

    test_df = find_sheet(
        sheets,
        target_name="test_questions",
        required_cols=["问题"]
    )
    if test_df is None:
        raise ValueError("未找到 test_questions 工作表，或缺少“问题”列。")

    test_df = test_df.fillna("")
    test_df.columns = [str(c).strip() for c in test_df.columns]

    results = []
    total = 0
    matched_count = 0
    exact_hit_count = 0

    for _, row in test_df.iterrows():
        total += 1

        qid = get_col(row, "id")
        question = get_col(row, "问题")
        # expected_id = get_col(row, "预期命中知识条目id", "预期知识条目id", "预期命中ID")
        expected_id_raw = get_col(row, "预期命中知识条目id", "预期知识条目id", "预期命中ID")
        expected_id = norm_id(expected_id_raw)
        campus = get_col(row, "校区", "适用校区")
        stage = get_col(row, "阶段", "适用阶段")
        should_fallback = get_col(row, "是否应触发兜底")

        item, score = search_knowledge(question, campus=campus, stage=stage)

        # matched = item is not None and score >= 3
        matched = item is not None and score >= MATCH_THRESHOLD
        if matched:
            matched_count += 1

        # predicted_id = item.get("id", "") if item else ""
        predicted_id = norm_id(item.get("id", "")) if item else ""
        predicted_question = item.get("标准问题", "") if item else ""
        predicted_cat1 = item.get("一级分类", "") if item else ""
        predicted_cat2 = item.get("二级分类", "") if item else ""

        exact_hit = False
        if expected_id:
            exact_hit = (predicted_id == expected_id)
            if exact_hit:
                exact_hit_count += 1

        results.append({
            "id": qid,
            "问题": question,
            "校区": campus,
            "阶段": stage,
            "预期命中知识条目id": expected_id,
            "预测知识条目id": predicted_id,
            "预测标准问题": predicted_question,
            "预测一级分类": predicted_cat1,
            "预测二级分类": predicted_cat2,
            "是否命中": "是" if matched else "否",
            "是否精确命中": "是" if exact_hit else "否" if expected_id else "",
            "匹配分数": round(score, 2) if item else 0,
            "是否应触发兜底": should_fallback,
        })
    labeled_count = sum(
        1 for row in results
        if str(row.get("预期命中知识条目id", "")).strip()
    )
    result_df = pd.DataFrame(results)

    matched_rate = matched_count / total if total else 0
    # exact_hit_rate = exact_hit_count / total if total else 0
    exact_hit_rate = exact_hit_count / labeled_count if labeled_count else 0


    summary_df = pd.DataFrame([
        {"指标": "测试总数", "数值": total},
        {"指标": "已标注预期ID题目数", "数值": labeled_count},
        {"指标": "成功返回结果数", "数值": matched_count},
        {"指标": "命中率", "数值": round(matched_rate, 4)},
        {"指标": "精确命中数", "数值": exact_hit_count},
        {"指标": "Top1精确命中率", "数值": round(exact_hit_rate, 4)},
    ])

    output_file = Path("evaluation_result.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        result_df.to_excel(writer, sheet_name="details", index=False)

    print("=" * 50)
    print("批量测试完成")
    print(f"测试总数: {total}")
    print(f"成功返回结果数: {matched_count}")
    print(f"命中率: {matched_rate:.2%}")
    print(f"精确命中数: {exact_hit_count}")
    print(f"Top1精确命中率: {exact_hit_rate:.2%}")
    print(f"结果文件: {output_file.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()