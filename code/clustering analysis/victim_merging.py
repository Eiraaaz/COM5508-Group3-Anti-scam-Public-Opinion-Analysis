
import pandas as pd
import json
import random
from zhipuai import ZhipuAI
from tqdm import tqdm
import time
# ================= Configuration =================
API_KEY = "508bd77a9a74430fb31e901c7ba60b37.Sedtu1p3fb75IIqt" 
client = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4-flash"  
BATCH_SIZE = 120            # 每批送入 LLM 的最大数量
MAX_ITERATIONS = 5          # 最多迭代轮数

INPUT_FILE = "victim_clustered_result.csv"
OUTPUT_FILE = "victim_semantic_merged.csv"
COLUMNS_TO_MERGE = [
    'victim_occupation_std',
    'victim_other_identity_std'
]
# ===============================================
def semantic_cluster_batch(values_batch, col_name, retries=2):
    if len(values_batch) <= 2:
        return {v: v for v in values_batch}
    values_str = "\n".join([f"- {v}" for v in values_batch])
    system_prompt = f"""You are an expert in data cleaning and semantic normalization.
I will give you a list of category labels from the column '{col_name}'. Some of these labels are **semantically identical or nearly identical** (e.g., "student" and "young student", "software engineer" and "software developer").
Your task is to **merge only the labels that refer to the same concept**, while keeping distinct categories separate. Do NOT merge different concepts just to reduce the number of categories.

RULES:
1. Merge labels if and only if they represent the same real-world entity or role.
2. If you are unsure, keep the original label.
3. The merged label should be a clear, concise representation of the group (preferably in English).
4. Output a JSON object mapping every original label (exactly as written) to its merged label.
5. Do NOT add any extra text or explanation.

OUTPUT FORMAT: Only a valid JSON object. Example:
{{"student": "Student", "young student": "Student", "teacher": "Teacher", "software engineer": "Software Developer", "software dev": "Software Developer"}}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": f"Labels to normalize:\n{values_str}"}],
                temperature=0.0,
                max_tokens=4095
            )
            res_raw = response.choices[0].message.content.strip()
            clean_json = res_raw.replace("```json", "").replace("```", "").strip()
            mapping = json.loads(clean_json)

            for v in values_batch:
                if v not in mapping:
                    mapping[v] = v
            return mapping
        except Exception as e:
            print(f" Try {attempt+1}/{retries} ERROR: {e}")
            time.sleep(2)
    
    return {v: v for v in values_batch}

#多轮迭代合并
def iterative_semantic_merge(unique_vals, col_name):
    current_vals = list(set(unique_vals))
    print(f"Initial Number: {len(current_vals)}")
    # 映射
    final_mapping = {v: v for v in unique_vals}
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f" ROUND {iteration}, current categories: {len(current_vals)}")
        random.shuffle(current_vals)

        batch_mappings = []
        for i in tqdm(range(0, len(current_vals), BATCH_SIZE), desc=f" Round {iteration} batches"):
            batch = current_vals[i:i+BATCH_SIZE]
            mapping = semantic_cluster_batch(batch, col_name)
            batch_mappings.append(mapping)

        round_mapping = {}
        for bmap in batch_mappings:
            round_mapping.update(bmap)

        new_final_mapping = {}
        for orig, prev_merged in final_mapping.items():
            new_merged = round_mapping.get(prev_merged, prev_merged)
            new_final_mapping[orig] = new_merged

        new_vals = list(set(new_final_mapping.values()))
        print(f" Round {iteration} final categories: {len(new_vals)}")
        final_mapping = new_final_mapping
        current_vals = new_vals
    return final_mapping


def main():
    print(f"INPUT: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')
        
    for col in COLUMNS_TO_MERGE:
        print(f"Processing: {col}")
        # 提取有效唯一值
        series = df[col].fillna("").astype(str).str.strip()
        valid_mask = ~series.isin(["", "Unknown", "None", "nan", "NaN"])
        unique_vals = series[valid_mask].unique().tolist()
        if len(unique_vals) <= 1:
            print(f"Too few valid unique values, skipped")
            df[f"{col}_merged"] = series.replace("", "Unknown")
            continue
        # 迭代语义合并
        mapping_dict = iterative_semantic_merge(unique_vals, col)
        def apply_map(x):
            if pd.isna(x) or str(x).strip() == "":
                return "Unknown"
            s = str(x).strip()
            return mapping_dict.get(s, s)
        merged_col = f"{col}_merged"
        df[merged_col] = df[col].apply(apply_map)

        final_cnt = df[merged_col].nunique()
        print(f"Completed. Final categories: {final_cnt} ")

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()