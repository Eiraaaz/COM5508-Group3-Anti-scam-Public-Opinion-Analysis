import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_excel("comments_cleaned_data_new.xlsx")
df["post_time"]=pd.to_datetime(df["post_time"])
df=df.dropna(subset=["post_time"])
df=df[~(df["post_time"].dt.date==pd.to_datetime("2025-10-24").date())]

daily=df.resample("D",on="post_time").size()
daily.name="count"
max_day=daily.idxmax()
max_val=daily.max()
threshold=max_val*0.5

# The public opinion lifecycle is divided into three periods based on temporal distribution rules:
# 1. Emergence Period: From the start of the event to the peak day, when public discussion rises continuously.
# 2. Peak Period: Days when comment volume reaches at least 50% of the maximum value, representing high public attention.
# 3. Cooling Period: After the peak, when comment volume drops below 50% of the peak, indicating fading public interest.
# This classification follows the standard law of public opinion evolution.
# 1. 高峰期：评论量>=峰值50%的所有日期
peak_mask=daily>=threshold
peak_days=daily[peak_mask].index
peak_start=peak_days.min()
peak_end=peak_days.max()
# 2. 爆发期：高峰期开始之前的所有日期
rise_mask=daily.index<peak_start
rise_days=daily[rise_mask].index
rise_start=rise_days.min()
rise_end=rise_days.max()
# 3. 冷静期： 高峰期结束之后的所有日期
cool_mask=daily.index>peak_end
cool_days=daily[cool_mask].index
cool_start=cool_days.min()
cool_end=cool_days.max()

if pd.isna(rise_start):
    rise_start=daily.index.min()
    rise_end=peak_start
if pd.isna(cool_start):
    cool_start=peak_end
    cool_end=daily.index.max()

plt.rcParams["font.sans-serif"]=["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(figsize=(12, 5))
plt.plot(daily.index,daily.values,color="#c72e29",linewidth=1.5,marker="o",markersize=3)
plt.axvspan(rise_start,rise_end,alpha=0.2,color="orange",label="Emergence Period")
plt.axvspan(peak_start,peak_end,alpha=0.2,color="red",label="Peak Period")
plt.axvspan(cool_start,cool_end,alpha=0.2,color="green",label="Cooling Period")
plt.annotate(f"Peak:{max_val}",
             xy=(max_day,max_val),
             xytext=(max_day,max_val+50),
             arrowprops=dict(arrowstyle="->",color="red"),
             fontsize=10,color="darkred")
plt.title("Comment Volume Time Curve & Public Opinion Evolution",fontsize=14)
plt.xlabel("Date",fontsize=12)
plt.ylabel("Number of Comments",fontsize=12)
plt.grid(True,linestyle="--",alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("comment_volume_with_three_periods.png", dpi=300)
plt.show()


print("="*50)
print("评论量时序分析结果")
print(f"峰值日期:{max_day.date()}")
print(f"峰值评论数:{max_val}")
print(f"爆发期:{rise_start.date()}—{rise_end.date()}")
print(f"高峰期:{peak_start.date()}—{peak_end.date()}")
print(f"冷静期:{cool_start.date()}—{cool_end.date()}")
print("="*50)
