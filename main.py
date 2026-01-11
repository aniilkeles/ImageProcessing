import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt


dataset_path = './Dataset'
img_size = (224, 224) 
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path, target_size=img_size, batch_size=batch_size,
    class_mode='categorical', subset='validation'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print("\nYeni veri seti ve MobileNetV2 ile eğitim başlıyor...")
history = model.fit(train_gen, validation_data=val_gen, epochs=15)


model.save('atik_modeli_final.h5')
print("\nModel 'atik_modeli_final.h5' adıyla kaydedildi.")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Test Başarısı')
plt.title('Doğruluk Oranı')
plt.legend()
plt.show()