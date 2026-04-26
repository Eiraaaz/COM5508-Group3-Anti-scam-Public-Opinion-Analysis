import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from openai import OpenAI

# ================= Configuration =================
API_KEY = "sk-TYSV5MHIjXLvokzoAsuuK0OYu7plTZfj9PsmI0aspZOY6ZJV" 
BASE_URL = "https://sg.uiuiapi.com/v1"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
MODEL_NAME = "gpt-4o-mini" 

INPUT_FILE = "comments_structured_result.csv" 
OUTPUT_FILE = "cluster_topics.csv" 
COLUMN_NAME = "main_theme_stage" 

INITIAL_CLUSTERS = 400
FINAL_MACRO_GOAL = "300"

EXCLUDED_VALUES = ["other", "N/A", "nan", ""] 

# ==========================================

def safe_api_call(messages, max_retries=5):
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000
            )
            return response
        except Exception as e:
            wait_time = (i + 1) * 2
            print(f"API Error...")
            time.sleep(wait_time)
    return None

def main():
    df = pd.read_csv(INPUT_FILE)
    df[COLUMN_NAME] = df[COLUMN_NAME].fillna("N/A").astype(str)
    print(f"Original data {len(df)} rows")

    is_excluded = df[COLUMN_NAME].str.strip().str.lower().isin(
        [v.strip().lower() for v in EXCLUDED_VALUES]
    )
    df_excluded = df[is_excluded].copy()
    df_to_cluster = df[~is_excluded].copy()

    if len(df_to_cluster) == 0:
        print("No topics available.")
        df['macro_topic'] = df[COLUMN_NAME]
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        return

    unique_topics = df_to_cluster[COLUMN_NAME].unique()
    unique_df = pd.DataFrame({COLUMN_NAME: unique_topics})
    print(f"Unique topics: {len(unique_df)}")

    embeddings_file = "cache_embeddings_filtered.npy"
    clustered_file = "cache_step1_clustered_filtered.csv"

    # ========== 向量化 ==========
    if not os.path.exists(embeddings_file):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        embeddings = model.encode(unique_df[COLUMN_NAME].tolist(), show_progress_bar=True, batch_size=64)
        np.save(embeddings_file, embeddings)
        print(f"Cached: {embeddings_file}")
    else:
        print(">>> Load cached vectors")
        embeddings = np.load(embeddings_file)

    # ========== K-Means 聚类 ==========
    if not os.path.exists(clustered_file):
        print(f">>> K-Means cluster ({INITIAL_CLUSTERS} groups)...")
        kmeans = KMeans(n_clusters=INITIAL_CLUSTERS, random_state=42, n_init=10)
        unique_df['temp_cluster_id'] = kmeans.fit_predict(embeddings)
        unique_df.to_csv(clustered_file, index=False, encoding='utf-8-sig')
        print(f"Initial clustering completed: {clustered_file}")
    else:
        print(">>> Load cached initial clustering")
        unique_df = pd.read_csv(clustered_file)

    # ========== 调用LLM ==========
    print(">>> Call LLM")

    cluster_names = {}
    grouped = unique_df.groupby('temp_cluster_id')[COLUMN_NAME].apply(list)

    sys_prompt = """You are an expert in anti-fraud and scam data analysis. 
I will provide a list of raw main themes extracted from scam-related comments. 

Your task: Analyze these themes and output a concise, specific scam category name in English (Max 6 words). 
The name should describe the exact scam type or tactic (e.g., "Investment Scam", "Phishing Link", "Impersonation Scam").

IMPORTANT RULES:
- Do NOT use vague terms like "Other", "Miscellaneous", "General", "Various", "Unknown Scam".
- Output must be a valid JSON object: {"name": "Category Name"}
"""

    for cid, items in tqdm(grouped.items(), total=len(grouped), desc="Naming clusters"):
        sample_str = str(items)
        if len(sample_str) > 5000:
            sample_str = sample_str[:5000] + "...[truncated]"

        response = safe_api_call([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Items: {sample_str}"}
        ])

        if response:
            try:
                res_json = json.loads(response.choices[0].message.content)
                name = res_json.get('name', f"Cluster_{cid}")
            except:
                name = f"Cluster_{cid}"
        else:
            name = f"Cluster_{cid}"

        if name.lower() in ["other", "miscellaneous", "others", "various", "unknown scam", "uncategorized"]:
            name = f"Cluster_{cid}"

        cluster_names[str(cid)] = name
        time.sleep(0.5)

    # ========== 宏观议题 ==========

    cluster_list_str = "\n".join([f"ID:{k} | {v}" for k, v in cluster_names.items()])

    final_sys_prompt = f"""You are an expert in public opinion analysis with a focus on fraud and scam detection.
I have a list of {len(cluster_names)} intermediate scam-related categories. 
Your task is to group them into {FINAL_MACRO_GOAL} final Macro Scam Topics.

**CRITICAL**: Your JSON output MUST include ALL {len(cluster_names)} cluster IDs as keys. Do not omit any ID.

IMPORTANT RULES:
- Every Macro Topic must be a specific fraud type or scam method.
- DO NOT use vague terms like "Other", "Miscellaneous", "General", "Various", "Uncategorized Scam".

Output a JSON where KEY is the intermediate category ID (string), VALUE is the Macro Topic name.
"""

    final_response = safe_api_call([
        {"role": "system", "content": final_sys_prompt},
        {"role": "user", "content": f"List:\n{cluster_list_str}"}
    ])

    if not final_response:
        print("ERROR.")
        return

    try:
        final_id_to_macro = json.loads(final_response.choices[0].message.content)
    except:
        final_id_to_macro = {}

    # ========== 自动补全缺失的簇ID ==========
    all_ids = set(cluster_names.keys())
    returned_ids = set(final_id_to_macro.keys())
    missing_ids = all_ids - returned_ids

    if missing_ids:
        for cid in missing_ids:
            final_id_to_macro[cid] = cluster_names.get(cid, f"Topic_{cid}")

    for cid, macro in final_id_to_macro.items():
        if macro.lower() in ["other", "miscellaneous", "others", "various", "unknown scam", "uncategorized"]:
            final_id_to_macro[cid] = cluster_names.get(cid, f"Topic_{cid}")

    print(f"Final macro topics: {len(set(final_id_to_macro.values()))}")

    # ========== 映射回原始数据 ==========
    topic_to_id = dict(zip(unique_df[COLUMN_NAME], unique_df['temp_cluster_id'].astype(str)))

    df_to_cluster['_temp_id'] = df_to_cluster[COLUMN_NAME].map(topic_to_id)
    df_to_cluster['macro_topic'] = df_to_cluster['_temp_id'].map(
        lambda x: final_id_to_macro.get(x, cluster_names.get(x, f"Topic_{x}")))
    df_to_cluster.drop(columns=['_temp_id'], inplace=True)

    df_excluded['macro_topic'] = df_excluded[COLUMN_NAME]

    df_final = pd.concat([df_to_cluster, df_excluded], ignore_index=True)
    df_final = df_final.sort_index().reset_index(drop=True)

    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"Completed.")

if __name__ == "__main__":
    main()