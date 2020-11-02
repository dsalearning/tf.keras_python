from keras.datasets import mnist
import matplotlib.pyplot as plt

# 載入 MNIST 資料集, 如果需要, 會自行下載
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 繪出9個數字圖片
sub_plot= 330
for i in range(0, 9):
    ax = plt.subplot(sub_plot+i+1)
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title("Label: " + str(Y_train[i]))
    ax.axis("off")

plt.subplots_adjust(hspace = .5)
# 顯示數字圖片
plt.show()
