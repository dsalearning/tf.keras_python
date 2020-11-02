import pandas as pd 

# 載入波士頓房屋資料集
df = pd.read_csv("./boston_housing.csv")

print(df.head())
df.head().to_html("./Ch5_4_1.html")
print(df.shape)

