import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

# 1. Veri yolu
data_dir = 'Rice_Image_Dataset'

# 2. Parametreler
img_size = (224, 224)
batch_size = 32
epochs = 10

# 3. Data Augmentation + validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# ✅ ÖNCE train_data'yı oluştur
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

# ✅ class_indices bozulmadan önce sınıf sayısını al
num_classes = len(train_data.class_indices)

# ✅ SONRA val_data'yı oluştur
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# 4. EfficientNetB0 modeli
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# 5. Derleme
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Eğitim
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[early_stop])

# 7. Kaydet
model.save("rice_transfer_model.h5")

# 8. Eğitim Grafikleri
st.subheader("Train vs Validation Accuracy / Loss")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

st.pyplot(plt.gcf())

# 9. Görsel Yükle ve Tahmin Et
st.subheader("Bir Görsel Yükle ve Tahmin Et")

uploaded_file = st.file_uploader("Bir görsel seç...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_input)
    class_idx = np.argmax(prediction)
    class_label = list(train_data.class_indices.keys())[class_idx]

    st.image(img, channels="BGR", caption="Yüklenen Görsel", width=250)
    st.success(f"Tahmin Edilen Sınıf: {class_label}")
