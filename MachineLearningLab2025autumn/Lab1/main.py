import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import ast
from datetime import datetime
import matplotlib.pyplot as plt

# 读取数据
features = pd.read_excel("Lab1/yii2/PR_features.xlsx")
print("features数据示例：")
print(features.head())

# 处理 pr_time_dict.txt
with open("Lab1/yii2/pr_time_dict.txt", "r", encoding="utf-8") as f:
    text = f.read()
    text = text.replace("nan", "None")
    pr_time_dict = ast.literal_eval(text)

# 将 pr_time_dict 转换为 DataFrame
dataset = pd.DataFrame.from_dict(pr_time_dict, orient="index")

# 将 pr_time_dict 的索引转换为列，方便与 features 合并
dataset.reset_index(inplace=True)
dataset.rename(columns={'index': 'number'}, inplace=True)

# 合并 features 和 dataset
merged_data = pd.merge(features, dataset, on='number', how='inner')

# 检查合并后的数据
print("合并后的数据示例：")
print(merged_data.head())

# 添加首个反馈时长 (TFR) 和关闭时长 (TTC) 列
merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])
merged_data['updated_at'] = pd.to_datetime(merged_data['updated_at'])
merged_data['closed_at'] = pd.to_datetime(merged_data['closed_at'])
merged_data['merged_at'] = pd.to_datetime(merged_data['merged_at'])

# 计算 TFR 和 TTC
merged_data['TFR'] = (merged_data['updated_at'] - merged_data['created_at']).dt.total_seconds() / 3600  # 转换为小时
merged_data['TTC'] = (merged_data[['merged_at', 'closed_at']].min(axis=1) - merged_data['created_at']).dt.total_seconds() / 3600  # 转换为小时

# 检查添加的列
print("添加 TFR 和 TTC 后的数据示例：")
print(merged_data[['number', 'TFR', 'TTC']].head())

merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])

train_df = merged_data[merged_data["created_at"] < "2021-06-01"]
test_df  = merged_data[merged_data["created_at"] >= "2021-06-01"]

print("训练集大小:", train_df.shape)
print("测试集大小:", test_df.shape)