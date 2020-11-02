from google.colab import drive
drive.mount("/content/drive")

import os
os.listdir("/content/drive/My Drive")

import pandas as pd

df = pd.read_csv("/content/drive/My Drive/iris.csv")
df.head(5)