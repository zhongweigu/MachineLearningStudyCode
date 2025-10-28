import re
import pandas as pd
import numpy as np
import ast
from datetime import datetime
import os
from tqdm import tqdm


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
    
base_dir = "Lab2/documents"

for project_name in tqdm(os.listdir(base_dir), desc="Processing projects", unit="project"):
    project_path = os.path.join(base_dir, project_name)

    # 只处理文件夹
    if not os.path.isdir(project_path):
        continue

    print(f"\n🔹 正在处理项目：{project_name}")

    feature_path = os.path.join(project_path, "PR_features.xlsx")
    time_dict_path = os.path.join(project_path, "pr_time_dict.txt")
    output_path = os.path.join(project_path, "processed_features.xlsx")

    if not os.path.exists(feature_path) or not os.path.exists(time_dict_path):
        print(f"⚠️ 跳过 {project_name}：文件缺失")
        continue

    # 读取数据
    features = pd.read_excel(feature_path)
    print("features数据示例：")
    print(features.head())

    # 处理 pr_time_dict.txt
    with open(time_dict_path, "r", encoding="utf-8") as f:
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


    print("合并后的数据示例：")
    print(merged_data.head())

    print("生成新的features......")
    merged_data.to_excel(output_path, index=False)
    print(f"新的features已保存到 {output_path}")