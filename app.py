import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# CSVファイルを読み込む
file_paths: dict[str, str] = {
    'Swallow 7B': 'swallow-7b.csv',
    'Swallow 13B': 'swallow-13b.csv',
    'Swallow 70B': 'swallow-70b.csv'
}

dataframes = {}

# Read each CSV file and convert iterations to tokens
for model, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    df['tokens'] = df['iteration'] * 4096 * 1024  # Convert iterations to tokens
    dataframes[model] = df

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the training loss for each model
for model, df in dataframes.items():
    plt.plot(df['tokens'], df['training-loss'], label=model)

# Set x-axis ticks to be at every 20 billion tokens
token_ticks = np.arange(0, max(df['tokens'].max() for df in dataframes.values()), 20e9)  # 20e9 tokens = 20B tokens
plt.xticks(token_ticks, [f"{int(tick/1e9)}B" for tick in token_ticks], fontsize=20)

# Adding labels, title, and grid
plt.xlabel('Billions of tokens', fontsize=22)  # X軸のラベルの文字サイズを設定
plt.ylabel('Training loss', fontsize=22)
# plt.title('Training loss for Swallow models over tokens')
plt.legend()
plt.grid(True)
plt.ylim(1.2, 3)
plt.xlim(0, 110e9)
plt.legend(fontsize=20)
plt.yticks(fontsize=20)

# Show the plot
plt.tight_layout()  # Adjust layout
plt.show()
