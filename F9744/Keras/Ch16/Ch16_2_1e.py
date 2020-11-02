from keras.models import Sequential
from keras.models import load_model

# 建立Keras的Sequential模型
model = Sequential()
model = load_model("imdb_gru.h5")
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="binary_crossentropy", optimizer="rmsprop", 
              metrics=["accuracy"])
# 顯示GRU層的權重形狀
print(2, model.layers[2].name, ":")
weights = model.layers[2].get_weights()
for i in range(len(weights)):
    print("==>", i, weights[i].shape)
