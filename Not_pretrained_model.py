import tensorflow as tf 
from keras.preprocessing.image import img_to_array, load_img 
import numpy as np 
import os
import matplotlib.pyplot as plt
import re

images_path = "C:/Users/surki/PycharmProjects/pythonProject15/images" 
masks_path = 'C:/Users/surki/PycharmProjects/pythonProject15/masks'

image_filenames = os.listdir(images_path) 
mask_filenames = os.listdir(masks_path)


images = [] 
masks = []
for image_name in image_filenames:
    # Извлекаем номер из названия файла
    image_num = re.findall(r'\d+', image_name)[0]
    
    # Загружаем изображение
    image = load_img(os.path.join(images_path, image_name), color_mode='rgb', target_size=(256, 256))
    image_array = img_to_array(image)
    images.append(image_array)
    
    # Находим маску с соответствующим номером
    mask_name = [mask_name for mask_name in mask_filenames if image_num in mask_name][0]
    
    mask = load_img(os.path.join(masks_path, mask_name), color_mode='grayscale', target_size=(256, 256))
    mask_array = img_to_array(mask)
    masks.append(mask_array)

images = np.array(images)
masks = np.array(masks)

model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)), 
    tf.keras.layers.MaxPooling2D(2, 2), 
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2, 2), 
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2, 2), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.Dense(256*256, activation='sigmoid'), 
    tf.keras.layers.Reshape((256, 256, 1))
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(images, masks, epochs=10, batch_size=32, validation_split=0.2) 
loss, accuracy = model.evaluate(images, masks)
print('Loss: ', loss, ' Accuracy: ', accuracy)

new_image_path = "C:/Users/surki/PycharmProjects/pythonProject15/OFT_control_01$000152&03_@000374.bmp"

new_image = load_img(new_image_path, color_mode='rgb', target_size=(256, 256))
new_image_array = img_to_array(new_image)
new_image_array = np.expand_dims(new_image_array, axis=0)

predicted_mask = model.predict(new_image_array)

# Визуализация результатов
plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
plt.show()