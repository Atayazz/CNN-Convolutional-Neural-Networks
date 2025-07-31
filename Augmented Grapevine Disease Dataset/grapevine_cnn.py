# grape_cnn.py

import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall

# Başlık
st.title("Grape Disease Detection")

# Parametreler
data_dir = "Final Training Data"
img_size = 128

# Veri yükleme
X, y = [], []
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        try:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(class_name)
        except:
            continue

X = np.array(X) / 255.0
y = np.array(y)

# Etiketleme
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Eğitim/Test verisi
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# CNN Modeli (5 Conv + 3 Pool)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

# Eğitim
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli Kaydet
model.save("grape_cnn_model.h5")

# Grafikler
st.subheader("Model Accuracy & Loss")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(history.history['accuracy'], label='Train Acc')
axs[0].plot(history.history['val_accuracy'], label='Val Acc')
axs[0].legend()
axs[0].set_title('Accuracy')

axs[1].plot(history.history['loss'], label='Train Loss')
axs[1].plot(history.history['val_loss'], label='Val Loss')
axs[1].legend()
axs[1].set_title('Loss')

st.pyplot(fig)

# Tahmin Bölümü
st.subheader("Bir Görsel Yükleyin ve Tahmin Edin")

uploaded_file = st.file_uploader("Bir görsel seç...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_input)
    predicted_class = le.inverse_transform([np.argmax(prediction)])

    st.image(img, channels="BGR", caption="Yüklenen Görsel", width=250)
    st.success(f"Tahmin Edilen Sınıf: {predicted_class[0]}")
