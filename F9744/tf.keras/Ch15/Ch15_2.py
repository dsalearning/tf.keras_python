from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

# 建立 MobileNet 模型
model = MobileNet(weights="imagenet", include_top=True) 
# 載入測試圖片
img = load_img("koala.png", target_size=(224, 224))
x = img_to_array(img)    # 轉換成 Numpy陣列
print("x.shape: ", x.shape)
# Reshape (1, 224, 224, 3)
img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
# 資料預處理
img = preprocess_input(img)
print("img.shape: ", img.shape)
# 使用模型進行預測
Y_pred = model.predict(img)
# 解碼預測結果
label = decode_predictions(Y_pred)
result = label[0][0]  # 取得最可能的結果
print("%s (%.2f%%)" % (result[1], result[2]*100))