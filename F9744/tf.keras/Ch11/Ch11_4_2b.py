from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU

# 定義模型
model = Sequential()
model.add(Embedding(10000, 32, input_length=100))
model.add(GRU(32, return_sequences=True))
model.add(GRU(32, return_sequences=True))
model.add(GRU(32))

model.summary()   # 顯示模型摘要資訊
