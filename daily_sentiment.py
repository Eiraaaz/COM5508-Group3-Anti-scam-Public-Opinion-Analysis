import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("comment_only_result.csv")
df["post_time"]=pd.to_datetime(df["post_time"],errors="coerce")
df=df.dropna(subset=["post_time"])

# 按天+情感分组，统计每日各情感的评论数量
daily_sentiment=df.groupby(
    [pd.Grouper(key="post_time",freq="D"), # 按天分组
     "sentiment"] # 按情感分组
).size().unstack().fillna(0) #没有数据的日期填写0
# 计算每日各情感占比
daily_total=df.resample("D",on="post_time").size()
daily_sentiment_pct=(daily_sentiment.div(daily_total,axis=0).fillna(0)*100).round(2)
plt.rcParams["font.sans-serif"]=["Arial Unicode MS"]  
plt.rcParams["axes.unicode_minus"]=False    

# 图1:每日各情感评论数量趋势
plt.figure(figsize=(12, 5))
for sentiment_type in daily_sentiment.columns:
    plt.plot(
        daily_sentiment.index, # x轴
        daily_sentiment[sentiment_type], # y轴
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=f"{sentiment_type}评论数"
    )

plt.title("Daily Sentiment Volume Time Series Trend",fontsize=14)
plt.xlabel("Date",fontsize=12)
plt.ylabel("Number of Comments",fontsize=12)
plt.grid(True,linestyle="--",alpha=0.3)  
plt.legend()  
plt.tight_layout()  
plt.savefig("daily_sentiment_volume_trend.png",dpi=300,bbox_inches="tight")  
plt.show()
# 图2：每日各情感评论占比趋势
plt.figure(figsize=(13, 6))
plt.stackplot(
    daily_sentiment_pct.index,
    daily_sentiment_pct["负面"],
    daily_sentiment_pct["中性"],
    daily_sentiment_pct["正面"],
    labels=["负面","中性","正面"],
    colors=["#E53E3E","#718096","#38A169"],
    alpha=0.75
)
plt.title("Sentiment Proportion Stacked Trend",fontsize=16)
plt.xlabel("Date",fontsize=12)
plt.ylabel("Proportion(%)",fontsize=12)
plt.ylim(0,100)
plt.legend(loc='upper left',fontsize=11) 
plt.xticks(rotation=30)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("sentiment_proportion_stacked_trend.png",dpi=300,bbox_inches="tight")
plt.show()


print("="*50)
print("每日情感评论数量统计:")
print(daily_sentiment)
print("\n每日情感评论占比统计:")
print(daily_sentiment_pct)
print("="*50)