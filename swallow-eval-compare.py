import matplotlib.pyplot as plt
import numpy as np

# モデルごとの性能データ
performance = [0.3098, 0.3201, 0.3940, 0.2923, 0.3963, 0.4625, 0.4830, 0.5528]
labels = ["calm2-7B", "Llama-2-7B", "Swallow-7B", "PLaMO-13b", "Llama-2-13B", "Swallow-13B", "Llama-2-70B", "Swallow-70B"]

# モデルごとに異なる色を指定
colors = ["tomato", "deepskyblue", "limegreen", "darkorange", "deepskyblue", "limegreen", "deepskyblue", "limegreen"]

# 棒グラフの位置設定、グループ間の空白を大きくする
x = np.arange(len(labels))
x[3:] += 1  # 13B の最初の棒グラフに1つ分のスペースを追加
x[6:] += 1  # 70B の最初の棒グラフにさらに1つ分のスペースを追加

fig, ax = plt.subplots(figsize=(12, 6))
rects = ax.bar(x, performance, color=colors)

# 軸ラベル、タイトルを追加
ax.set_ylabel("Performance")
ax.set_title("Performance by model")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)


# 棒グラフ上に数値を追加する関数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects)

fig.tight_layout()

plt.show()
