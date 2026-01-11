import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('atik_modeli_final.h5')


class_names = ['Cardboard', 'Glass', 'Metal', 'Paper','Plastic']


cap = cv2.VideoCapture(0)

print("Kamera açılıyor... Tahmin ekranı gelecektir.")
print("Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor!")
        break

   
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0) 

    
    prediction = model.predict(img, verbose=0)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index] * 100

  
    if confidence > 50:
        label = f"{class_names[result_index]}: %{confidence:.2f}"
        color = (0, 255, 0)
    else:
        label = "Analiz ediliyor..."
        color = (0, 165, 255) 

   
    cv2.rectangle(frame, (5, 5), (450, 65), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imshow('Final Projesi - Atik Siniflandirma', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()