import pandas as pd
import numpy as np

seed = 7
np.random.seed(seed)
# 載入資料集
df = pd.read_csv("./titanic_data.csv")
# 刪除不需要的欄位
df = df.drop(["name", "ticket", "cabin"], axis=1)
# 處理遺失資料
df[["age"]] = df[["age"]].fillna(value=df[["age"]].mean())
df[["fare"]] = df[["fare"]].fillna(value=df[["fare"]].mean())
df[["embarked"]] = df[["embarked"]].fillna(value=df["embarked"].
                   value_counts().idxmax())
print(df["embarked"].value_counts())
print(df["embarked"].value_counts().idxmax())
# 轉換分類資料
df["sex"] = df["sex"].map( {"female": 1, "male": 0} ).astype(int)
# Embarked欄位的One-hot編碼
enbarked_one_hot = pd.get_dummies(df["embarked"], prefix="embarked")
df = df.drop("embarked", axis=1)
df = df.join(enbarked_one_hot)
# 將標籤的 survived 欄位移至最後
df_survived = df.pop("survived") 
df["survived"] = df_survived
print(df.head())
df.head().to_html("Ch6_2_2.html")
# 分割成訓練(80%)和測試(20%)資料集
mask = np.random.rand(len(df)) < 0.8
df_train = df[mask]
df_test = df[~mask]
print("Train:", df_train.shape)
print("Test:", df_test.shape)
# 儲存處理後的資料
df_train.to_csv("titanic_train.csv", index=False)
df_test.to_csv("titanic_test.csv", index=False)