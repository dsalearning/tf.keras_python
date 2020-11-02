import pandas as pd

df = pd.read_csv("./diabetes.csv")

print(df.head())
df.head().to_html("./Ch5_2_1.html")
print(df.shape)

