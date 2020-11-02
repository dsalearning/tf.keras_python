from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
# 載入圖檔
img = load_img("penguins.png")
# 顯示圖片資訊
print(type(img))
# 轉換成 Numpy 陣列
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)
# 將 Numpy 陣列轉換成 Image
img2 = array_to_img(img_array)
print(type(img2))
# 顯示圖片
import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(img2)