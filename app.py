import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# CSVファイルを読み込む
data = pd.read_csv('swallow-13b-llm-jp.csv')

# x軸とy軸のデータを処理する
data['x'] = data['iteration'] / 250
data['y'] = data['training-loss']  # yのデータはそのまま使用

# 補間関数を作成
f = interp1d(data['x'], data['y'], kind='cubic')

# 新しいx軸のポイントを作成
new_x = np.linspace(data['x'].min(), data['x'].max(), num=1000, endpoint=True)

# プロットの作成
plt.figure(figsize=(10, 6))
plt.plot(new_x, f(new_x), color='#3599db', linestyle='-', linewidth=2)  # 青色の線でプロット

# 軸の範囲の設定
plt.xlim(0, 25000 / 250 + 2)
plt.ylim(1.3, 2.5)

# 軸のラベルを設定
plt.xlabel('Billon of tokens', fontsize=15)
plt.ylabel('Training loss', fontsize=15)

# グラフの表示
plt.show()
