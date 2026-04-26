# 1. 安装依赖
# !pip install openai pandas matplotlib requests threading

# 2. 导入核心库
import os
import re
import json
import time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------
# 配置参数
GLM_API_KEY = "a7188721eb134e37bd52ca6cf0a7d24c.v2hqUt13BJaN0kOH"  
COMMENT_FILE_PATH = "/Users/chaiyuxin/Desktop/comments_cleaned_data.csv"  # comment表路径
RESULT_SAVE_PATH = "/Users/chaiyuxin/Desktop/comment_only_result.csv"  # 结果保存路径

# --------------------------
# 模型与线程配置
BATCH_SIZE = 20
THREAD_NUM = 5
API_INTERVAL = 0.4
MODEL_NAME = "glm-4-flash"
MAX_TOKENS = 2000

# 初始化客户端
client = OpenAI(
    api_key=GLM_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# --------------------------
def analyze_batch(batch, batch_id):
    # 从批量数据中提取 文本、来源、时间 三个字段
    texts = [item[0] for item in batch]
    sources = [item[1] for item in batch]
    post_times = [item[2] for item in batch]  

    # 构造提示文本（仅用文本内容进行分析）
    prompt_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

    # 系统提示词
    system_prompt = """
你是专业情感分析专家，**只输出标准JSON数组**，不输出任何多余文字、解释、换行、备注。
每条格式严格为：
{"review":"原文","language":"普通话/英语/粤语","sentiment":"正面/负面/中性","confidence":0.95,"post_time":"原始时间"}
必须是完整的JSON数组，例如：
[{"review":"你好","language":"普通话","sentiment":"中性","confidence":0.9,"post_time":"2025/12/28 4:56"}]
"""
    try:
        time.sleep(API_INTERVAL)  # 控制API调用频率
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"分析以下内容：\n{prompt_text}"}
            ],
            temperature=0,  # 固定温度，保证结果稳定性
            max_tokens=MAX_TOKENS
        )

        # 提取并解析JSON结果
        content = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', content, re.DOTALL)  # 匹配JSON数组
        if not match:
            print(f"❌ 批量{batch_id} 未提取到JSON结果")
            return []

        results = json.loads(match.group(0))
        # 为每条结果添加 source 和 post_time 字段
        for i, res in enumerate(results):
            if i < len(sources):
                res["source"] = sources[i]  # 来源标识（comment）
                res["post_time"] = post_times[i]  # 添加时间字段
        print(f"✅ 批量{batch_id} 完成：{len(results)} 条记录")
        return results

    except Exception as e:
        print(f"❌ 批量{batch_id} 处理失败：{str(e)[:50]}")  # 捕获并简化错误信息
        return []


# --------------------------
def load_comment_only():
    all_data = []
    print("📥 正在读取评论数据（含post_time字段）...")
    
    # 分块读取CSV，避免内存占用过高
    df_chunks = pd.read_csv(COMMENT_FILE_PATH, chunksize=10000)
    for chunk in df_chunks:
        # 过滤空值并转换数据类型
        valid_data = chunk[["clean_content", "post_time"]].dropna()  # 同时读取两个字段
        valid_data["clean_content"] = valid_data["clean_content"].astype(str)
        valid_data["post_time"] = valid_data["post_time"].astype(str)  # 保持时间格式原样
        
        # 构建三元组列表：(文本内容, 来源, 时间)
        chunk_list = [
            (row["clean_content"], "comment", row["post_time"])
            for _, row in valid_data.iterrows()
        ]
        all_data.extend(chunk_list)
    
    print(f"✅ 数据读取完成：共 {len(all_data)} 条有效记录")
    return all_data

# --------------------------
# 多线程执行函数（逻辑不变）
# --------------------------
def run_fast():
    all_data = load_comment_only()
    # 按批次分割数据
    batches = [all_data[i:i+BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"\n🚀 开始情感分析：共 {total_batches} 个批次，每批次 {BATCH_SIZE} 条")

    all_results = []
    # 多线程执行批量分析
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        # 提交所有批次任务
        futures = [
            executor.submit(analyze_batch, batch, batch_id+1)
            for batch_id, batch in enumerate(batches)
        ]
        # 收集完成的任务结果
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)

    # 保存结果到CSV
    df_result = pd.DataFrame(all_results)
    df_result.to_csv(RESULT_SAVE_PATH, index=False, encoding="utf-8-sig")
    
    # 输出执行总结
    print(f"\n🎉 全部分析完成！")
    print(f"📊 统计信息：")
    print(f"   - 总输入记录数：{len(all_data)}")
    print(f"   - 成功分析数：{len(all_results)}")
    print(f"   - 结果保存路径：{RESULT_SAVE_PATH}")
    print(f"   - 结果字段：{df_result.columns.tolist()}")

# --------------------------
# 启动执行
# --------------------------
if __name__ == "__main__":
    run_fast()
