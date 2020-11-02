from tensorflow.keras.datasets import imdb

# 載入 IMDb 資料集, 如果是第一次載入會自行下載資料集
top_words = 1000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
                                num_words=top_words)
# 形狀
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)
# 顯示 Numpy 陣列內容
print(X_train[0])
print(Y_train[0])   # 標籤資料

