
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取处理好的数据
print("读取处理好的数据...")
merged_data = pd.read_excel("Lab1/yii2/processed_features.xlsx")

# 添加首个反馈时长 (TFR) 和关闭时长 (TTC) 列
merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])
merged_data['updated_at'] = pd.to_datetime(merged_data['updated_at'])
merged_data['closed_at'] = pd.to_datetime(merged_data['closed_at'])
merged_data['merged_at'] = pd.to_datetime(merged_data['merged_at'])
merged_data = merged_data.dropna(subset=["closed_at"])

# 计算 TFR 和 TTC
merged_data['TFR'] = (merged_data['updated_at'] - merged_data['created_at']).dt.total_seconds() / 3600  # 转换为小时
merged_data['TTC'] = (merged_data[['merged_at', 'closed_at']].min(axis=1) - merged_data['created_at']).dt.total_seconds() / 3600  # 转换为小时

merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])

drop_cols = ["modify_proportion","modify_entropy"]
merged_data = merged_data.drop(columns=drop_cols, errors="ignore")

ignore_cols = ["TFR", "TTC","created_at","number","updated_at","closed_at","merged_at"]
cols_to_fill = [c for c in merged_data.columns if c not in ignore_cols]

merged_data[cols_to_fill] = merged_data[cols_to_fill].fillna(0)

# 检查合并后的数据
print("数据示例：")
print(merged_data.head())

train_df = merged_data[merged_data["created_at"] < "2021-06-01"]
test_df  = merged_data[merged_data["created_at"] >= "2021-06-01"]

train_df = train_df.dropna(subset=["TTC"])
test_df  = test_df.dropna(subset=["TTC"])

print("训练集大小:", train_df.shape)
print("测试集大小:", test_df.shape)

# 目标
y_train = train_df["TTC"]
y_test  = test_df["TTC"]

# 特征列（排除掉无关的）
drop_cols = ["number", "created_at", "updated_at", "merged_at", "closed_at", "TFR", "TTC","modify_proportion","modify_entropy"]
X_train = train_df.drop(columns=drop_cols, errors="ignore")
X_test  = test_df.drop(columns=drop_cols, errors="ignore")
print("特征维度:", X_train.shape[1])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


print("训练模型...")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} 结果：")
    print(f"  MAE = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R² = {r2:.2f}")
    print("-" * 40)
