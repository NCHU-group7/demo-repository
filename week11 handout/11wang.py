import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 輸入檔案路徑
train_file = "week11 handout/train.csv"  # 請填寫訓練集檔案路徑
test_file = "week11 handout/test.csv"  # 請填寫測試集檔案路徑

# 讀取資料
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# 查看前 10 筆資料
print("訓練集前 10 筆資料：")
print(train_data.head(10))

# 資料預處理
def preprocess_data(data):
    # 填補缺失值
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # 類別標籤轉換
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    
    # 特徵標準化
    scaler = StandardScaler()
    features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
    data[features] = scaler.fit_transform(data[features])
    
    return data

# 預處理數據
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 提取特徵與目標變數
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_data['Survived']

# 將訓練數據拆分為訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 混淆矩陣可視化
def plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if title == "Logistic Regression" else 'Greens', cbar=True, ax=ax)
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")

# 模型訓練與評估
def train_and_evaluate(model, model_name, ax):
    model.fit(X_train, y_train)  # 在訓練集上訓練
    y_pred_val = model.predict(X_val)  # 在驗證集上進行預測
    
    print(f"\n{model_name} 驗證集評估指標：")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.2f}")
    print(f"Precision: {precision_score(y_val, y_pred_val):.2f}")
    print(f"Recall: {recall_score(y_val, y_pred_val):.2f}")
    print(f"F1-score: {f1_score(y_val, y_pred_val):.2f}")
    
    # 畫混淆矩陣
    plot_confusion_matrix(y_val, y_pred_val, model_name, ax)

# 建立圖表框架
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 羅吉斯迴歸
logistic_model = LogisticRegression()
train_and_evaluate(logistic_model, "Logistic Regression", axes[0])

# 決策樹
decision_tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
train_and_evaluate(decision_tree_model, "Decision Tree", axes[1])

# 顯示圖表
plt.tight_layout()
plt.show()