import pandas as pd

# 載入資料集
df = pd.read_csv("./titanic_data.csv")
# 顯示資料集的資訊
print(df.info())
# 顯示沒有資料的筆數
print(df.isnull().sum())