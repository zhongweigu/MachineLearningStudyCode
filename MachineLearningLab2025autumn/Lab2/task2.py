import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 🔧 定义 MLP 分类网络
# -----------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 🧩 读取多个项目数据
# -----------------------------
base_dir = "Lab2/documents"
all_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}")

print("📂 正在读取所有项目 processed_features.xlsx ...")
for project in tqdm(os.listdir(base_dir), desc="Loading projects", unit="proj"):
    project_path = os.path.join(base_dir, project, "processed_features.xlsx")
    if os.path.exists(project_path):
        df = pd.read_excel(project_path)
        df["project"] = project  # 标记项目名
        all_data.append(df)
    else:
        print(f"⚠️ 跳过 {project}，未找到 processed_features.xlsx")

merged_data = pd.concat(all_data, ignore_index=True)
print(f"✅ 共读取 {len(all_data)} 个项目，总样本数: {merged_data.shape[0]}")

# -----------------------------
# 🕒 时间字段处理与标签生成
# -----------------------------
for col in ["created_at", "updated_at", "closed_at", "merged_at"]:
    merged_data[col] = pd.to_datetime(merged_data[col], errors="coerce")

# 只保留关闭的 PR
merged_data = merged_data.dropna(subset=["closed_at"])

# 标签：是否合并（merged_at 非空）
merged_data["is_merged"] = merged_data["merged_at"].notna().astype(int)

# -----------------------------
# 🧹 数据清洗
# -----------------------------
drop_cols = ["number", "created_at", "updated_at", "closed_at", "merged_at"]
ignore_cols = drop_cols + ["is_merged", "project"]
merged_data = merged_data.replace([np.inf, -np.inf], 0)
merged_data = merged_data.fillna(0)

# -----------------------------
# ✂️ 时间分割
# -----------------------------
print("✂️ 正在划分训练集与测试集...")

train_df = merged_data[merged_data["created_at"] < "2021-06-01"]
test_df  = merged_data[
    (merged_data["created_at"] >= "2021-06-01") &
    (merged_data["created_at"] <= "2022-06-15")
]

X_train = train_df.drop(columns=ignore_cols, errors="ignore").values
y_train = train_df["is_merged"].values

X_test  = test_df.drop(columns=ignore_cols, errors="ignore").values
y_test  = test_df["is_merged"].values

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# -----------------------------
# ⚙️ 标准化
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"✅ 特征维度: {X_train.shape[1]}")

# -----------------------------
# 🚀 构建与训练模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

model = MLPClassifier(X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()

epochs = 50
print("🚀 开始训练神经网络分类器...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(train_dl)
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] 平均Loss: {avg_loss:.6f}")

print("✅ 训练完成！")

# -----------------------------
# 🧮 模型评估
# -----------------------------
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_t.to(device)).cpu().numpy().flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n📊 测试结果：")
print(f"Accuracy  = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall    = {rec:.4f}")
print(f"F1        = {f1:.4f}")
print("-" * 40)
