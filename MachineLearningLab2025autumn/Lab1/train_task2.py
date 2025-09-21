# train_merge.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取处理好的数据
print("读取处理好的数据...")
merged_data = pd.read_excel("Lab1/yii2/processed_features.xlsx")

# 时间字段处理
merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])
merged_data['closed_at'] = pd.to_datetime(merged_data['closed_at'])
merged_data['merged_at'] = pd.to_datetime(merged_data['merged_at'])

# 只保留已经关闭的 PR（即要么合并要么拒绝）
merged_data = merged_data.dropna(subset=["closed_at"])

# 生成标签列：是否合并
merged_data['is_merged'] = merged_data['merged_at'].notna().astype(int)

# 丢掉不需要的列
drop_cols = ["number", "created_at", "updated_at", "closed_at", "merged_at"]
X = merged_data.drop(columns=drop_cols, errors="ignore")
y = merged_data["is_merged"]

# 缺失值填充
X = X.fillna(0)

# 时间切分：和任务一保持一致
train_df = merged_data[merged_data["created_at"] < "2021-06-01"]
test_df  = merged_data[merged_data["created_at"] >= "2021-06-01"]

X_train = train_df.drop(columns=drop_cols + ["is_merged"], errors="ignore").fillna(0)
X_test  = test_df.drop(columns=drop_cols + ["is_merged"], errors="ignore").fillna(0)
y_train = train_df["is_merged"]
y_test  = test_df["is_merged"]

print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 缺失值填充
X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

# 处理 inf 值
X_train = X_train.replace([np.inf, -np.inf], 0)
X_test  = X_test.replace([np.inf, -np.inf], 0)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 定义模型
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 训练和评估
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"{name} 结果：")
    print(f"  Accuracy = {acc:.4f}")
    print(f"  Precision = {prec:.4f}")
    print(f"  Recall = {rec:.4f}")
    print(f"  F1 = {f1:.4f}")
    print("-" * 40)
