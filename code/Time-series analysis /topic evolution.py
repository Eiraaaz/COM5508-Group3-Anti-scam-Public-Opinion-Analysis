import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("cluster_topics.csv")
df["post_time"]=pd.to_datetime(df["post_time"])
df=df.dropna(subset=["post_time"])
df["topic_label"]=df["clustered_topic"]
daily_topic=df.groupby([
    pd.Grouper(key="post_time",freq="D"),
    "topic_label"
]).size().unstack().fillna(0)

daily_total=daily_topic.sum(axis=1)
daily_topic_pct = daily_topic.div(daily_total,axis=0)*100
max_day=daily_total.idxmax()
max_val=daily_total.max()
threshold=max_val*0.5

burst_start=daily_total.index.min()
burst_end=max_day
peak_start=daily_total[daily_total>=threshold].index.min()
peak_end=daily_total[daily_total>=threshold].index.max()
cool_start=peak_end
cool_end=daily_total.index.max()

# 图1:主题热度曲线
plt.rcParams["font.sans-serif"]=["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(figsize=(15,7))

for topic in daily_topic.columns:
    plt.plot(daily_topic.index,daily_topic[topic],marker="o",markersize=2,label=topic)

plt.axvspan(burst_start,burst_end,alpha=0.2,color="orange",label="Emergence Period")
plt.axvspan(peak_start,peak_end,alpha=0.2,color="red",label="Peak Period")
plt.axvspan(cool_start,cool_end,alpha=0.2,color="green",label="Cooling Period")
plt.title("Topic Evolution Over Time",fontsize=14)
plt.xlabel("Date",fontsize=12)
plt.ylabel("Number of Comments",fontsize=12)
plt.grid(True,linestyle="--",alpha=0.3)
plt.legend().remove()
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.savefig("topic_trend.png",dpi=300,bbox_inches="tight")
plt.show()
# 图2:主题迁移堆叠面积图
plt.figure(figsize=(14, 6))
daily_topic_pct.plot.area(stacked=True,alpha=0.7,ax=plt.gca())

plt.axvspan(burst_start,burst_end,alpha=0.1,color="orange")
plt.axvspan(peak_start,peak_end,alpha=0.1,color="red")
plt.axvspan(cool_start,cool_end,alpha=0.1,color="green")
plt.title("Topic Migration Stacked Trend",fontsize=14)
plt.xlabel("Date",fontsize=12)
plt.ylabel("Proporation",fontsize=12)
plt.legend().remove()
plt.subplots_adjust(bottom=0.1, top=0.9)
plt.savefig("topic_migration_stack.png",dpi=300,bbox_inches="tight")
plt.show()


burst_df=df[df["post_time"]<=burst_end]
peak_df=df[(df["post_time"]>=peak_start)&(df["post_time"]<=peak_end)]
cool_df=df[df["post_time"]>=cool_start]
print("="*60)
print("爆发期 热门主题 TOP3")
print(burst_df["topic_label"].value_counts().head(3))
print("\n高峰期 热门主题 TOP3")
print(peak_df["topic_label"].value_counts().head(3))
print("\n冷静期 热门主题 TOP3")
print(cool_df["topic_label"].value_counts().head(3))
print("="*60)
