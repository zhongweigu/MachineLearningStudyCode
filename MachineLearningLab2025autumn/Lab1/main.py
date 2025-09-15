import re
import pandas as pd
import numpy as np
import ast
from datetime import datetime


def parse_space_array(s):
    if isinstance(s, str):
        # 去掉首尾的中括号和多余空格
        s = s.strip().strip("[]")
        # 按空格切分（可能有多个空格，所以用正则）
        parts = re.split(r"\s+", s)
        try:
            return [float(x) for x in parts if x != ""]
        except ValueError:
            return []
    elif isinstance(s, (list, np.ndarray)):
        return list(s)
    else:
        return []

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

# # 检查合并后的数据
# print("合并后的数据示例：")
# print(merged_data.head())

# # 添加首个反馈时长 (TFR) 和关闭时长 (TTC) 列
# merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])
# merged_data['updated_at'] = pd.to_datetime(merged_data['updated_at'])
# merged_data['closed_at'] = pd.to_datetime(merged_data['closed_at'])
# merged_data['merged_at'] = pd.to_datetime(merged_data['merged_at'])

# # 计算 TFR 和 TTC
# merged_data['TFR'] = (merged_data['updated_at'] - merged_data['created_at']).dt.total_seconds() / 3600  # 转换为小时
# merged_data['TTC'] = (merged_data[['merged_at', 'closed_at']].min(axis=1) - merged_data['created_at']).dt.total_seconds() / 3600  # 转换为小时


array_cols = ['title_embedding', 'body_embedding', 'comment_embedding']

for col in array_cols:
    if col not in merged_data.columns:
        print(f"⚠️ 跳过不存在的列: {col}")
        continue

    # 转换为 list[float]
    merged_data[col] = merged_data[col].apply(parse_space_array)

    # 拆开成多列
    expanded = pd.DataFrame(
        merged_data[col].tolist(),
        index=merged_data.index
    )

    # 重命名列
    expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]

    # 合并回去
    merged_data = pd.concat([merged_data.drop(columns=[col]), expanded], axis=1)

merged_data.to_excel("Lab1/yii2/processed_features.xlsx", index=False)




# # for col in merged_data.columns:
# #     if merged_data[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
# #         print("删除列:", col)
# #         merged_data = merged_data.drop(columns=[col])





# # 检查添加的列
# print("添加 TFR 和 TTC 后的数据示例：")
# print(merged_data[['number', 'TFR', 'TTC']].head())

# merged_data['created_at'] = pd.to_datetime(merged_data['created_at'])

# train_df = merged_data[merged_data["created_at"] < "2021-06-01"]
# test_df  = merged_data[merged_data["created_at"] >= "2021-06-01"]

# train_df = train_df.dropna(subset=["TTC"])
# test_df  = test_df.dropna(subset=["TTC"])

# print("训练集大小:", train_df.shape)
# print("测试集大小:", test_df.shape)

# # 目标
# y_train = train_df["TTC"]
# y_test  = test_df["TTC"]

# # 特征列（排除掉无关的）
# drop_cols = ["number", "created_at", "updated_at", "merged_at", "closed_at", "TFR", "TTC","modify_proportion","modify_entropy"]
# X_train = train_df.drop(columns=drop_cols, errors="ignore")
# X_test  = test_df.drop(columns=drop_cols, errors="ignore")
# print("特征维度:", X_train.shape[1])



# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)

# models = {
#     "LinearRegression": LinearRegression(),
#     "Ridge": Ridge(alpha=1.0),
#     "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
# }

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"{name} 结果：")
#     print(f"  MAE = {mae:.2f}")
#     print(f"  RMSE = {rmse:.2f}")
#     print(f"  R² = {r2:.2f}")
#     print("-" * 40)
