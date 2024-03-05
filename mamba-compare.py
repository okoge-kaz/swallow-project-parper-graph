import matplotlib.pyplot as plt
import numpy as np

# モデルごとの日本語と英語の性能データ
performance_ja = [0.1772, 0.1819, 0.2089, 0.2123]  # 日本語の性能
performance_en = [0.3991, 0.1821, 0.2812, 0.3901]  # 英語の性能
labels = [
    "open-llama-3b_v2",
    "open-calm-3b",
    "mamba-2.8b (kotomamba)",
    "mamba-2.8b (kotomamba-CL)",
]

# モデルごとに異なる色を指定（日本語と英語で色を変える）
colors_ja = ["tomato"] * len(performance_ja)  # 日本語の性能の色
colors_en = ["deepskyblue"] * len(performance_en)  # 英語の性能の色

# 棒グラフの位置設定
x = np.arange(len(labels))  # ラベルの数だけ位置を生成

# 棒グラフの幅
bar_width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))

# 日本語の性能の棒グラフ
rects_ja = ax.bar(x - bar_width / 2, performance_ja, bar_width, label="Japanese", color="tomato")

# 英語の性能の棒グラフ
rects_en = ax.bar(x + bar_width / 2, performance_en, bar_width, label="English", color="deepskyblue")

# 軸ラベル、タイトル、凡例を追加
ax.set_ylabel("Performance")
ax.set_title("Performance by model and language")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()


# 棒グラフ上に数値を追加する関数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",  # 小数点以下4桁でフォーマット
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


autolabel(rects_ja)
autolabel(rects_en)

fig.tight_layout()

plt.show()
