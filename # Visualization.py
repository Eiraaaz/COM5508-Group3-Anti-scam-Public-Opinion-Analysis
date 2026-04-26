# 1. 安装依赖（若未安装，先运行这行）
# !pip install pandas matplotlib numpy datetime

# 2. 导入库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 3. 配置参数（只需修改文件路径）
# --------------------------
# 你的情感分析结果文件路径
FILE_PATH = "/Users/chaiyuxin/Desktop/comment_only_result.csv"  # 替换成你的文件实际路径
# 图表保存路径（可自定义）
SAVE_PATH = "/Users/chaiyuxin/Desktop/Sentiment_Analysis_Visualization"
# 设置中文字体（解决中文显示乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --------------------------
# 4. 数据加载与预处理 + 标签英文化
# --------------------------
print("Loading data...")
df = pd.read_csv(FILE_PATH)

# 数据清洗：去除空值，格式化时间
df = df.dropna(subset=['post_time', 'sentiment', 'language'])
# 转换时间格式（适配 "2025/12/28 2:37" 这类格式）
df['post_time'] = pd.to_datetime(df['post_time'], format='%Y/%m/%d %H:%M', errors='coerce')
# 提取日期（按天聚合）
df['date'] = df['post_time'].dt.date
# 过滤无效数据
df = df[df['sentiment'].isin(['正面', '负面', '中性'])]
df = df[df['language'].isin(['普通话', '英语', '粤语'])]

# 🔄 关键：将中文标签替换为英文
df['language'] = df['language'].map({
    '普通话': 'Mandarin',
    '英语': 'English',
    '粤语': 'Cantonese'
})
df['sentiment'] = df['sentiment'].map({
    '正面': 'Positive',
    '负面': 'Negative',
    '中性': 'Neutral'
})

print(f"Data loaded successfully! Total {len(df)} valid records")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
print(f"Language distribution:\n{df['language'].value_counts()}")

# --------------------------
# 5. 生成多维度可视化图表（全英文）
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comment Sentiment Analysis Report', fontsize=20, fontweight='bold', y=0.95)

# 定义配色方案（专业美观）
colors_sentiment = {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#4682B4'}
colors_language = {'Mandarin': '#FF6347', 'English': '#32CD32', 'Cantonese': '#9370DB'}

# ———— 子图1：情感分布饼图 ————
sentiment_counts = df['sentiment'].value_counts()
wedges, texts, autotexts = axes[0,0].pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    colors=[colors_sentiment[label] for label in sentiment_counts.index],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 11}
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
axes[0,0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold', pad=20)

# ———— 子图2：语言分布柱状图 ————
language_counts = df['language'].value_counts()
bars = axes[0,1].bar(
    language_counts.index,
    language_counts.values,
    color=[colors_language[label] for label in language_counts.index],
    alpha=0.8,
    edgecolor='black',
    linewidth=1
)
# 在柱状图上添加数值标签
for bar in bars:
    height = bar.get_height()
    axes[0,1].text(
        bar.get_x() + bar.get_width()/2., height + 5,
        f'{int(height)}',
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )
axes[0,1].set_title('Language Distribution', fontsize=14, fontweight='bold', pad=20)
axes[0,1].set_ylabel('Number of Comments', fontsize=12)
axes[0,1].grid(axis='y', alpha=0.3)

# ———— 子图3：时间趋势图（按天统计情感变化） ————
time_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
# 确保所有情感类型都存在
for sentiment in ['Positive', 'Negative', 'Neutral']:
    if sentiment not in time_sentiment.columns:
        time_sentiment[sentiment] = 0
# 绘制折线图
for sentiment in ['Positive', 'Negative', 'Neutral']:
    axes[1,0].plot(
        time_sentiment.index,
        time_sentiment[sentiment],
        label=sentiment,
        color=colors_sentiment[sentiment],
        linewidth=2.5,
        marker='o',
        markersize=4
    )
axes[1,0].set_title('Daily Sentiment Trend', fontsize=14, fontweight='bold', pad=20)
axes[1,0].set_xlabel('Date', fontsize=12)
axes[1,0].set_ylabel('Number of Comments', fontsize=12)
axes[1,0].legend(loc='upper right', fontsize=10)
axes[1,0].grid(alpha=0.3)
# 旋转x轴日期标签
axes[1,0].tick_params(axis='x', rotation=45)

# ———— 子图4：语言-情感交叉热力图 ————
cross_table = pd.crosstab(df['language'], df['sentiment'])
im = axes[1,1].imshow(
    cross_table.values,
    cmap='YlOrRd',
    aspect='auto'
)
# 添加数值标签
for i in range(len(cross_table.index)):
    for j in range(len(cross_table.columns)):
        axes[1,1].text(
            j, i, cross_table.iloc[i,j],
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='white' if cross_table.iloc[i,j] > cross_table.values.max()/2 else 'black'
        )
# 设置坐标轴标签
axes[1,1].set_xticks(range(len(cross_table.columns)))
axes[1,1].set_xticklabels(cross_table.columns, fontsize=11)
axes[1,1].set_yticks(range(len(cross_table.index)))
axes[1,1].set_yticklabels(cross_table.index, fontsize=11)
axes[1,1].set_title('Language-Sentiment Cross Distribution', fontsize=14, fontweight='bold', pad=20)
# 添加颜色条
cbar = plt.colorbar(im, ax=axes[1,1], shrink=0.8)
cbar.set_label('Number of Comments', fontsize=10)

# --------------------------
# 6. 调整布局并保存图表
# --------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.92)
# 创建保存目录（若不存在）
import os
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
# 保存图表（高分辨率）
plt.savefig(
    f"{SAVE_PATH}/Sentiment_Analysis_Report.png",
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()

# --------------------------
# 7. 生成补充图表：各语言情感占比堆叠柱状图
# --------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# 计算各语言的情感占比
language_sentiment = pd.crosstab(df['language'], df['sentiment'], normalize='index') * 100
# 堆叠柱状图
language_sentiment.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=[colors_sentiment[col] for col in language_sentiment.columns],
    alpha=0.8,
    edgecolor='black',
    linewidth=1
)
ax.set_title('Sentiment Distribution by Language (Stacked Bar)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Language', fontsize=14)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.legend(title='Sentiment', fontsize=12, title_fontsize=12)
ax.grid(axis='y', alpha=0.3)
# 在柱子上添加数值标签
for i, language in enumerate(language_sentiment.index):
    bottom = 0
    for sentiment in language_sentiment.columns:
        value = language_sentiment.loc[language, sentiment]
        if value > 5:  # 只显示占比>5%的标签
            ax.text(
                i, bottom + value/2,
                f'{value:.1f}%',
                ha='center', va='center',
                fontsize=10, fontweight='bold'
            )
        bottom += value
# 保存补充图表
plt.tight_layout()
plt.savefig(
    f"{SAVE_PATH}/Sentiment_Distribution_by_Language.png",
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()

print(f"\n✅ All charts saved to: {SAVE_PATH}")
print("Generated charts:")
print("1. Sentiment_Analysis_Report.png (Comprehensive 4-subplot report)")
print("2. Sentiment_Distribution_by_Language.png (Stacked bar chart by language)")