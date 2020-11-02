import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料集
df = pd.read_csv("./iris_data.csv")
target_mapping = {"setosa": 0,
                  "versicolor": 1,
                  "virginica": 2}
Y = df["target"].map(target_mapping)
# 使用Matplotlib顯示視覺化圖表
colmap = np.array(["r", "g", "y"])
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.subplots_adjust(hspace = .5)
plt.scatter(df["sepal_length"], df["sepal_width"], color=colmap[Y])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.subplot(1, 2, 2)
plt.scatter(df["petal_length"], df["petal_width"], color=colmap[Y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# 使用Seaborn顯示視覺化圖表
sns.pairplot(df, hue="target")
