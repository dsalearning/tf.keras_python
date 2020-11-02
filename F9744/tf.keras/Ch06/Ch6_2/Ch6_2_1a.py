import pandas as pd

# 載入資料集
df = pd.read_csv("./titanic_data.csv")
# 查看前5筆記錄
print(df.head())
df.head().to_html("Ch6_2_1a_01.html")
# 顯示資料集的描述資料
print(df.describe())
df.describe().to_html("Ch6_2_1a_02.html")