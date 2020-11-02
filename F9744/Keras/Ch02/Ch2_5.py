import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(10)  # 指定亂數種子
# 建立測試資料
X = np.linspace(-1, 1, 300)
np.random.shuffle(X)  # 使用亂數打亂資料
Y = 0.3 * X + 5 + np.random.normal(0, 0.05, (300, ))
# 繪出資料的散佈圖
plt.scatter(X, Y)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()

X_train, Y_train = X[:270], Y[:270]     # 訓練資料前270點
X_test, Y_test = X[270:], Y[270:]       # 測試資料後30點

# 建立Keras的Sequential模型
model = Sequential()
model.add(Dense(2, input_dim=1))        # 隱藏層 2個神經元
model.add(Dense(1))                     # 輸出層 1個神經元
# 編譯模型
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
model.summary()

# 訓練模型
print("Training ....")
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=30)

# 顯示訓練和驗證損失
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# 測試模型
print("\nTesting ...")
cost = model.evaluate(X_test, Y_test, batch_size=30)
print("Test Cost =", cost)
print("\nHidden Layer ...")
W, b = model.layers[0].get_weights()
print("Weights=", W, "\nbiases=", b)  
print("\nOutput Layer ...")
W, b = model.layers[1].get_weights()
print("Weights=", W, "\nbiases=", b)

# 預測模型
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.xlabel("X")
plt.ylabel("Y")

plt.show()