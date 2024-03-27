import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import markers
from matplotlib.pylab import f

# データの準備
data = {
    "iteration": [0, 5000, 10000, 15000, 20000, 25000],
    "Swallow-7B": [0.3201, 0.3604, 0.3725, 0.3799, 0.3910, 0.3940],
    "Swallow-13B": [0.3963, 0.4335, 0.4408, 0.4472, 0.4579, 0.4625],
    "Swallow-70B": [0.4847, 0.5236, 0.5371, 0.5452, 0.5482, 0.5528],
}

# DataFrameの作成
df = pd.DataFrame(data)
df["Billions of tokens"] = df["iteration"] * 4096 * 1024 / 1e9

# 表の作成
plt.figure(figsize=(10, 6))
plt.plot(df["Billions of tokens"], df["Swallow-7B"], label="Llama 2-JA-7b", marker="o")
plt.plot(df["Billions of tokens"], df["Swallow-13B"], label="Llama 2-JA-13b", marker="o")
plt.plot(df["Billions of tokens"], df["Swallow-70B"], label="Llama 2-JA-70b", marker="o")

# グラフのタイトルと軸ラベルの設定
plt.xlabel("Billions of tokens", fontsize=26, fontweight=900)
plt.ylabel("Japanese Score (mean)", fontsize=26, fontweight=900)
plt.grid(True)
plt.ylim(0.30, 0.57)
plt.xlim(-1.5, 110)
plt.legend(fontsize=20)
plt.yticks(fontsize=20, fontweight=900)
plt.xticks(fontsize=20, fontweight=900)

plt.tight_layout()
plt.show()
