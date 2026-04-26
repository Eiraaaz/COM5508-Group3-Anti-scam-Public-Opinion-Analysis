
import pandas as pd
import json
import ast
import time
from tqdm import tqdm
from zhipuai import ZhipuAI

# ========== Configuration ==========
API_KEY = "d7ee3d85476c4e08a2eb3f8db49322f8.kj5giXUDp95FJinZ"
client = ZhipuAI(api_key=API_KEY)
MODEL_NAME = "glm-4-flash"

INPUT_FILE = "comments_structured_result.csv"
OUTPUT_FILE = "scam_analysis_clustered.csv"
COLUMN_LABEL = "fraud_types"

CHUNK_SIZE = 200          # 第一次归纳每批标签数
FINAL_MAP_BATCH = 200     # 第三次映射每批标签数
MAX_CATEGORIES = 40       # 最终类别数量上限

IGNORE_LIST_LOWER = ["none", "无", "api_error", "error_retry", "[]", "null", ""]

# ============================
def parse_multiple_labels(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    val_str = str(val).strip()
    if val_str.lower() in IGNORE_LIST_LOWER:
        return []
    try:
        parsed = ast.literal_eval(val_str)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
        return [val_str]
    except (ValueError, SyntaxError):
        return [val_str]


def call_llm(system_prompt, user_content, temperature=0.1, max_tokens=4095):
    for attempt in range(3): 
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw = response.choices[0].message.content
            
            clean = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except Exception as e:
            print(f"⚠️ LLM 调用失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(2)
    return {}

# ========== 第一遍：粗归纳 ==========
# 初步类别列表
def rough_clustering(all_labels):
    rough_categories = set()
    print(f"First rough clusteing,  {len(all_labels)} unique tag, {CHUNK_SIZE} per batch")
    system_prompt = (
        "You are a fraud analysis expert. You will receive a list of raw scam type labels extracted from social media.\n"
        "Your task: Group these labels into coherent categories and output a JSON object where:\n"
        "- Each key is a category name (short and clear, in English)\n"
        "- Each value is a brief description (1 sentence) of what that category covers.\n\n"
        "Requirements:\n"
        "1. Output ONLY a valid JSON object, no extra text.\n"
        "2. Do NOT create too many categories; try to keep it under 15 per batch.\n"
        "3. Category names should be consistent across batches (e.g., use 'Investment Scam', not sometimes 'Investment Fraud').\n"
        "4. Example format: {\"Investment Scam\": \"Schemes promising high returns through fake trading platforms.\", ...}"
    )
    for i in tqdm(range(0, len(all_labels), CHUNK_SIZE), desc="粗归纳"):
        batch = all_labels[i:i+CHUNK_SIZE]
        user_content = "Raw labels:\n" + "\n".join([f"- {lbl}" for lbl in batch])
        result = call_llm(system_prompt, user_content, temperature=0.1)
        if result:
            rough_categories.update(result.keys())
        
        time.sleep(0.5)
    return list(rough_categories)

# ========== 第二遍：精归纳 ==========
def refine_categories(rough_list):
    print(f"Second Round clusteing, Merge {len(rough_list)} rough categories.")
    system_prompt = (
        "You are a fraud taxonomy expert. You will receive a list of category names (some may be duplicates or similar).\n"
        f"Your task: Merge similar categories and produce a final list of NO MORE THAN {MAX_CATEGORIES} distinct categories.\n"
        "For each final category, provide a clear name and a brief description.\n\n"
        "Output ONLY a JSON object where keys are final category names and values are descriptions.\n"
        "Ensure the names are concise and consistent (prefer 'Impersonation Scam' over 'Pretending to be someone else')."
    )
    user_content = "Rough category list:\n" + "\n".join([f"- {c}" for c in rough_list])
    final_cat_dict = call_llm(system_prompt, user_content, temperature=0.0, max_tokens=3000)
    if not final_cat_dict:
        final_cat_dict = {c: f"Auto-generated from rough: {c}" for c in rough_list[:MAX_CATEGORIES]}

    if len(final_cat_dict) > MAX_CATEGORIES:
        final_cat_dict = dict(list(final_cat_dict.items())[:MAX_CATEGORIES])
    print(f"Final {len(final_cat_dict)} categories:  ")
    for name, desc in final_cat_dict.items():
        print(f"  - {name}: {desc[:50]}...")
    return final_cat_dict

# ========== 第三遍：标签映射 ==========
def map_labels_to_final(all_labels, final_categories_dict):
    master_map = {}
    category_names = list(final_categories_dict.keys())
    category_desc = "\n".join([f"- {name}: {desc}" for name, desc in final_categories_dict.items()])
    system_prompt = (
        "You are a data labeling assistant. You will be given:\n"
        "1) A fixed list of scam categories with descriptions.\n"
        "2) A batch of raw scam labels.\n\n"
        "Your task: For each raw label, assign it to EXACTLY ONE category from the fixed list (choose the best fit).\n"
        "Output ONLY a JSON object where keys are the exact raw labels and values are the chosen category name.\n"
        "If a label truly does not fit any category, map it to 'Other Scam'.\n"
        "Example output: {\"fake investment\": \"Investment Scam\", \"earn money typing\": \"Employment/Task Scam\"}"
    )
    print(f"Label mapping,  {len(all_labels)} tags in total.")
    for i in tqdm(range(0, len(all_labels), FINAL_MAP_BATCH), desc="Mapping"):
        batch = all_labels[i:i+FINAL_MAP_BATCH]
        user_content = f"FIXED CATEGORIES:\n{category_desc}\n\nRAW LABELS TO MAP:\n" + "\n".join([f"- {lbl}" for lbl in batch])
        batch_map = call_llm(system_prompt, user_content, temperature=0.0, max_tokens=3000)
        if batch_map:
            master_map.update(batch_map)
        else:
            for lbl in batch:
                master_map[lbl] = "Other Scam"
        time.sleep(0.5)

    for lbl in all_labels:
        if lbl not in master_map:
            master_map[lbl] = "Other Scam"
    return master_map

# ===========================
def main():
    print(f"INPUT: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='gbk')
    # 提取所有唯一有效标签
    df['parsed_fraud_types'] = df[COLUMN_LABEL].apply(parse_multiple_labels)
    all_raw = []
    for lst in df['parsed_fraud_types']:
        all_raw.extend(lst)
    unique_labels = list(set([lbl for lbl in all_raw if lbl and lbl.lower() not in IGNORE_LIST_LOWER]))
    print(f"Valid Unique Tags: {len(unique_labels)}")

    # 2. 第一遍粗归纳 → 得到粗类别列表
    rough_cats = rough_clustering(unique_labels)
    print(f"Rough categories: {len(rough_cats)}")

    # 3. 第二遍精归纳 → 得到最终类别体系
    final_categories = refine_categories(rough_cats)

    # 4. 第三遍映射 → 得到标签到最终类别的映射表
    label_to_category = map_labels_to_final(unique_labels, final_categories)

    # 5. 应用到每行数据
    def map_row(label_list):
        if not label_list:
            return ["Other Scam"]
        mapped = []
        for lbl in label_list:
            if lbl.lower() in IGNORE_LIST_LOWER:
                continue
            mapped.append(label_to_category.get(lbl, "Other Scam"))
        unique_mapped = list(set(mapped))
        return unique_mapped if unique_mapped else ["Other Scam"]

    df['main_scam_category_list'] = df['parsed_fraud_types'].apply(map_row)
    df['main_scam_category'] = df['main_scam_category_list'].apply(lambda x: json.dumps(x, ensure_ascii=False))

    df.drop(columns=['parsed_fraud_types', 'main_scam_category_list'], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()