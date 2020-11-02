from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
# 載入圖檔
img = load_img("penguins.png", grayscale=True)
# 顯示圖片資訊
print(type(img))
# 轉換成 Numpy 陣列
img_array = img_to_array(img)
# 儲存圖檔
save_img("penguins_grayscale.jpg", img_array)
# 載入圖片
img2 = load_img("penguins_grayscale.jpg")
# 顯示圖片
import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(img2, cmap="gray")

