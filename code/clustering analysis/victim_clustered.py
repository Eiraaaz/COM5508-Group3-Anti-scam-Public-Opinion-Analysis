import pandas as pd
import json
import random
from zhipuai import ZhipuAI
from tqdm import tqdm

# ================= Configuration =================
API_KEY = "508bd77a9a74430fb31e901c7ba60b37.Sedtu1p3fb75IIqt"  
client = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4-flash"

INPUT_FILE = "comments_structured_result.csv"   
OUTPUT_FILE = "victim_clustered_result.csv"     

VICTIM_COLS = [
    'victim_gender',
    'victim_age_group',
    'victim_occupation',
    'victim_other_identity'
]

# 期望生成的主类别数量
VICTIM_MASTER_COUNTS = {
    'victim_gender': 3,
    'victim_age_group': 12,
    'victim_occupation': 100,
    'victim_other_identity': 400,
}

# ===============================================

# 阶段一：生成全局主类别列表
def generate_master_categories_victim(unique_vals, col_name, num_master):
    sample_size = min(800, len(unique_vals))
    sample = random.sample(unique_vals, sample_size)
    values_str = "\n".join([f"- {t}" for t in sample])

    system_prompt = f"""You are a anti-fraud expert specializing in victim profiling.

TASK: Extract the **global, unified {num_master} main categories** from the following provided original labels: {col_name}.

NON-NEGOTIABLE RULES:
1. Limit the number of categories to {num_master} and ensure that many similar concepts are consolidated.
2. The category name should be concise and professional (preferably in English, consisting of 2 to 6 words).
3. Avoid the categories “Miscellaneous”, “Other”, and “Unknown”.
4. Output format: **Returns only a valid JSON array.**"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"Raw labels for {col_name}:\n{values_str}"}],
            temperature=0.1,
            max_tokens=2048
        )
        res_raw = response.choices[0].message.content.strip()
        clean_json = res_raw.replace("```json", "").replace("```", "").strip()
        master_list = json.loads(clean_json)
        print(f"    ✅ 【{col_name}】阶段1生成 {len(master_list)} 个主类别")
        return master_list
    except Exception as e:
        print(f"生成 {col_name} 主类别失败: {e}")
        return []

#阶段二：将原始标签映射到主类别
def map_to_master_victim(chunk, master_categories, col_name):
    values_str = "\n".join([f"- {t}" for t in chunk])
    master_str = "\n".join([f"- {cat}" for cat in master_categories])

    system_prompt = f"""You are a world-class anti-fraud expert.
OFFICIAL CATEGORIES(choose from these strictly; do not create new ones):
{master_str}

RULES:
- The only option available is the one that best matches from the list above.
- It is strictly forbidden to create new categories or to return to 'Other'.
- Output format: Only valid JSON objects will be returned."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"Raw labels to map:\n{values_str}"}],
            temperature=0.0,
            max_tokens=4095
        )
        res_raw = response.choices[0].message.content
        clean_json = res_raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"API Error: {e}")
        return {}


def main():
    print(f"INPUT: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')

    print(f"Original Data ROWS: {len(df)}")

    # 对每个受害者字段进行两阶段聚类标准化
    for orig_col in VICTIM_COLS:

        std_col = f"{orig_col}_std"
        num_master = VICTIM_MASTER_COUNTS[orig_col]
        print(f"Processing: {orig_col}")
        # ===== 去重+过滤=====
        raw_series = df[orig_col].fillna("").astype(str).str.strip()
        unique_vals = raw_series.unique().tolist()
        unique_vals = [v for v in unique_vals if v and len(v) > 1]
        print(f"The only value: {len(unique_vals)}")

        # 如果数据已经比较干净，直接用原始值
        if len(unique_vals) <= num_master + 5:
            df[std_col] = raw_series.replace("", "Unknown")
            continue

        # 阶段一：生成主类别
        master_cats = generate_master_categories_victim(unique_vals, orig_col, num_master)
        if len(master_cats) == 0:
            print(f"ERROR, revert to using the original value.")
            df[std_col] = raw_series.replace("", "Unknown")
            continue

        # 阶段二：映射
        mapping = {}
        batch_size = 300
        for i in tqdm(range(0, len(unique_vals), batch_size), desc=f"   映射 {orig_col}"):
            chunk = unique_vals[i:i+batch_size]
            chunk_mapping = map_to_master_victim(chunk, master_cats, orig_col)
            mapping.update(chunk_mapping)

        def apply_map(v):
            if pd.isna(v) or str(v).strip() == "":
                return "Unknown"
            s = str(v).strip()
            return mapping.get(s, s)

        df[std_col] = df[orig_col].apply(apply_map)
        final_categories = df[std_col].nunique()
        print(f"Final categories: {final_categories}")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # Statistics
    print("Categories in each dimension：")
    for col in VICTIM_COLS:
        std_col = f"{col}_std"
        if std_col in df.columns:
            print(f"   {std_col}: {df[std_col].nunique()} 类")

if __name__ == "__main__":
    main()