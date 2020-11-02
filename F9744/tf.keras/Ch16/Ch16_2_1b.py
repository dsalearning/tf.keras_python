from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# 建立Keras的Sequential模型
model = Sequential()
model = load_model("mnist.h5")
model.summary()   # 顯示模型摘要資訊
# 編譯模型
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 顯示各層的權重形狀
for i in range(len(model.layers)):
    print(i, model.layers[i].name, ":")
    weights = model.layers[i].get_weights()
    for j in range(len(weights)):
        print("==>", j, weights[j].shape)