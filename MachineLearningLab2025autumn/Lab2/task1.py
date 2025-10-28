import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm   # ✅ 新增进度条库

# ==============================
# 一、定义神经网络模型
# ==============================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# 二、读取所有项目 processed_features.xlsx
# ==============================
base_dir = "Lab2/documents"
all_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}")

print("📂 正在读取所有项目数据...")
project_dirs = [p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))]

for project in tqdm(project_dirs, desc="读取项目", unit="project"):
    project_path = os.path.join(base_dir, project, "processed_features.xlsx")
    if os.path.exists(project_path):
        df = pd.read_excel(project_path)
        df["project"] = project
        all_data.append(df)
    else:
        tqdm.write(f"⚠️ 跳过 {project}：文件不存在")

merged_data = pd.concat(all_data, ignore_index=True)
print(f"\n✅ 共读取 {len(all_data)} 个项目，总数据量: {merged_data.shape}\n")


# ==============================
# 三、时间处理 & 特征工程
# ==============================
print("🕒 正在处理时间特征...")

for col in tqdm(["created_at", "updated_at", "closed_at", "merged_at"], desc="转换时间列"):
    merged_data[col] = pd.to_datetime(merged_data[col], errors="coerce")

merged_data = merged_data.dropna(subset=["closed_at"])

merged_data["TFR"] = (merged_data["updated_at"] - merged_data["created_at"]).dt.total_seconds() / 3600
merged_data["TTC"] = (merged_data[["merged_at", "closed_at"]].min(axis=1) - merged_data["created_at"]).dt.total_seconds() / 3600

drop_cols = ["modify_proportion", "modify_entropy"]
merged_data = merged_data.drop(columns=drop_cols, errors="ignore")

ignore_cols = ["TFR", "TTC", "created_at", "updated_at", "closed_at", "merged_at", "number", "project"]
cols_to_fill = [c for c in merged_data.columns if c not in ignore_cols]

print("🧹 填充缺失值...")
for col in tqdm(cols_to_fill, desc="填充缺失值", unit="col"):
    merged_data[col] = merged_data[col].fillna(0)


# ==============================
# 四、数据集划分（时间分割）
# ==============================
print("\n✂️ 正在划分训练集与测试集...")

train_df = merged_data[merged_data["created_at"] < "2021-06-01"]
test_df  = merged_data[
    (merged_data["created_at"] >= "2021-06-01") & 
    (merged_data["created_at"] <= "2022-06-15")
]

train_df = train_df.dropna(subset=["TTC"])
test_df  = test_df.dropna(subset=["TTC"])

print(f"训练集大小: {train_df.shape}, 测试集大小: {test_df.shape}")


# ==============================
# 五、特征与标签准备
# ==============================
y_train = train_df["TTC"].values
y_test  = test_df["TTC"].values

drop_cols = ["number", "created_at", "updated_at", "merged_at", "closed_at", "TFR", "TTC", "modify_proportion", "modify_entropy", "project"]
X_train = train_df.drop(columns=drop_cols, errors="ignore").values
X_test  = test_df.drop(columns=drop_cols, errors="ignore").values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"✅ 特征维度: {X_train.shape[1]}")


# ==============================
# 六、PyTorch数据加载
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 使用设备: {device}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)


# ==============================
# 七、训练模型（含进度条）
# ==============================
model = MLPRegressor(X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

epochs = 100
print("\n🚀 开始训练神经网络...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    # ✅ 每个 epoch 内部的 batch 进度条
    with tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False) as batch_bar:
        for xb, yb in batch_bar:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] 平均Loss: {total_loss/len(train_dl):.4f}")

print("\n✅ 训练完成！")


# ==============================
# 八、测试评估
# ==============================
print("\n🧮 正在进行模型评估...")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_t.to(device)).cpu().numpy().flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 测试结果：")
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")
