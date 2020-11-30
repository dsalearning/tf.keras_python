## 6-2 實作案例 : 鐵達尼號資料集的生存分析

import pandas as pd
import numpy as np

# 指定亂數種子
np.random.seed(5)

# 載入資料集
df = pd.read_csv("./titanic_data.csv")

# 檢驗資料
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# 資料預處理
# 1. name, ticket, cabin 跟是否生存沒有關係，所以刪除這三欄
# 2. sex 需要mapping成數字
# 3. age, fare, embarked 都有遺失值，以平均值填補
# 4. embarked 需要用 one-hot encoding 轉成分類欄位

# 1. 刪除不用的欄位
df = df.drop(["name","ticket","cabin"],axis=1)
print(df.head())

# 2. sex 對應, 0 male; 1 female
df["sex"] = df["sex"].map({"male":0,"female":1}).astype(int)
print(df.head())

# 3. 遺失值處理
# age, fare 以平均值填入
df["age"] = df["age"].fillna(value=df["age"].mean())
df["fare"] = df["fare"].fillna(value=df["fare"].mean())
# embarked 以Max Count的值填入
df["embarked"] = df["embarked"].fillna(value=df["embarked"].value_counts().idxmax())
print(df.describe())

# 4. embarked的One-hot encoding
embarked_one_hot = pd.get_dummies(df["embarked"],prefix="embarked")
df = df.drop("embarked",axis=1)
df = df.join(embarked_one_hot)
print(df.head())

# 將標籤資料移到最後
df_survived = df.pop("survived")
df["survived"] = df_survived
print(df.head())

# 分割成訓練資料與測試資料
mask = np.random.rand(len(df))< 0.8
df_train = df[mask]
df_test = df[~mask]
print("Train : ", df_train.shape)
print("Test : ",df_test.shape)

# 儲存成訓練和測試資料集的 CSV 檔案
df_train.to_csv("titanic_train.csv", index=False)
df_test.to_csv("titanic_test.csv", index=False)
