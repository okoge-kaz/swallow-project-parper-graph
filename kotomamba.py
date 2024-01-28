import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('science')

# CSVファイルを読み込む
file_paths: dict[str, str] = {
    'kotomamba-2.8b': 'kotomamba.csv',
}

dataframes = {}

# Read each CSV file and convert iterations to tokens
for model, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    df['tokens'] = df['iteration'] * 1024 * 2048  # Convert iterations to tokens
    dataframes[model] = df

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the training loss for each model
for model, df in dataframes.items():
    plt.plot(df['tokens'], df['training-loss'], label=model)

# Set x-axis ticks to be at every 20 billion tokens
token_ticks = np.arange(0, max(df['tokens'].max() for df in dataframes.values()), 25e9)  # 20e9 tokens = 20B tokens
plt.xticks(token_ticks, [f"{int(tick/1e9)}B" for tick in token_ticks], fontsize=20)

# Adding labels, title, and grid
plt.xlabel('Billions of tokens', fontsize=22)  # X軸のラベルの文字サイズを設定
plt.ylabel('Training loss', fontsize=22)
# plt.title('Training loss for Swallow models over tokens')
plt.legend()
plt.grid(True)
plt.ylim(1.2, 6)
plt.xlim(0, 150e9)
plt.legend(fontsize=20)
plt.yticks(fontsize=20)

# Show the plot
plt.tight_layout()  # Adjust layout
plt.show()
