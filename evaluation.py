import matplotlib.pyplot as plt
import pandas as pd

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
plt.plot(df["Billions of tokens"], df["Swallow-7B"], label="Swallow-7b", marker="o", linewidth=3)
plt.plot(df["Billions of tokens"], df["Swallow-13B"], label="Swallow-13b", marker="o", linewidth=3)
plt.plot(df["Billions of tokens"], df["Swallow-70B"], label="Swallow-70b", marker="o", linewidth=3)

# グラフのタイトルと軸ラベルの設定
plt.xlabel("Billions of tokens", fontsize=30)
plt.ylabel("Japanese Score (mean)", fontsize=30)
plt.grid(True)
plt.ylim(0.25, 0.60)
plt.xlim(-1.5, 110)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)

# 凡例の位置を右下に設定
plt.legend(fontsize=24, loc='lower right', bbox_to_anchor=(1, -0.03))

# 0B tokensと100B tokensのスコアを表示
plt.text(2, df["Swallow-7B"].iloc[0] + 0.02, f'{df["Swallow-7B"].iloc[0]:.4f}', fontsize=15, va='bottom', ha='left')
plt.text(2, df["Swallow-13B"].iloc[0] + 0.02, f'{df["Swallow-13B"].iloc[0]:.4f}', fontsize=15, va='bottom', ha='left')
plt.text(2, df["Swallow-70B"].iloc[0] + 0.02, f'{df["Swallow-70B"].iloc[0]:.4f}', fontsize=15, va='bottom', ha='left')

plt.text(103, df["Swallow-7B"].iloc[-1], f'{df["Swallow-7B"].iloc[-1]:.4f}', fontsize=15, va='bottom', ha='right')
plt.text(103, df["Swallow-13B"].iloc[-1], f'{df["Swallow-13B"].iloc[-1]:.4f}', fontsize=15, va='bottom', ha='right')
plt.text(103, df["Swallow-70B"].iloc[-1], f'{df["Swallow-70B"].iloc[-1]:.4f}', fontsize=15, va='bottom', ha='right')

plt.tight_layout()
plt.show()
