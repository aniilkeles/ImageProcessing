import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = tf.keras.models.load_model('atik_modeli_final.h5')


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    './Dataset', 
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical', 
    subset='validation', 
    shuffle=False
)


print("Model test ediliyor, matris hazırlanıyor...")
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)


cm = confusion_matrix(val_gen.classes, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=list(val_gen.class_indices.keys()),
            yticklabels=list(val_gen.class_indices.keys()))

plt.title('Atık Sınıflandırma Karmaşıklık Matrisi')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Modelin Tahminleri')
plt.show()


print("\nSınıflandırma Raporu:\n")
print(classification_report(val_gen.classes, y_pred, target_labels=list(val_gen.class_indices.keys())))