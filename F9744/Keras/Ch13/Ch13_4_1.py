from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

img = load_img("koala.png") 
x = img_to_array(img) 
x = x.reshape((1,) + x.shape)  # reshape (1, 707, 505, 3)
print(x.shape)

datagen = ImageDataGenerator(
           rotation_range=40,
           width_shift_range=0.2,
           height_shift_range=0.2,
           shear_range=0.2,
           zoom_range=0.2,
           horizontal_flip=True)
i = 0
for batch_img in datagen.flow(x, batch_size=1,
                              save_to_dir="preview", 
                              save_prefix="pen",
                              save_format="jpeg"):
    plt.axis("off")
    plt.imshow(batch_img[0].astype("int"))
    plt.show()
    i += 1
    if i >= 10:
        break 

