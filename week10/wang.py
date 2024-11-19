import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv('week10.csv', delimiter=';')

# 顯示前 5 列和所有欄位類型
print(df.head())
print(df.info())

# 檢查是否有任何缺失值
print(df.isnull().sum())

# 針對每個特徵繪製 'y' 的計數長條圖
for feature in ['age', 'job', 'marital', 'education', 'loan']:
    # 計算每個類別中的 'yes' 和 'no' 的數量
    counts = df.groupby([feature, 'y']).size().reset_index(name='counts')

    # 建立長條圖
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, y_value in enumerate(['yes', 'no']):
        df_y = counts[counts['y'] == y_value]
        ax.bar(df_y[feature], df_y['counts'], label=y_value, alpha=0.6)

    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.set_title(f'{feature} vs y')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()