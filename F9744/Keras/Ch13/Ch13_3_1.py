from keras.preprocessing.image import load_img
# 載入圖檔
img = load_img("penguins.png")
# 顯示圖片資訊
print(type(img))
print(img.format)
print(img.mode)
print(img.size)
# 顯示圖片
import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(img)

