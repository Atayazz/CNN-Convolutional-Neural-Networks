import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st  # 🔸 EKLENDİ
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 1. Veri yolu
train_dir = 'archive/train'
val_dir = 'archive/validation'
test_dir = 'archive/test'

# 2. Parametreler
img_size = (224, 224)
batch_size = 32
epochs = 1

# 3. Veri hazırlama
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_data = val_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# 4. EfficientNetB0 modeli
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Eğitim
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[early_stop])

# 7. Kaydet
model.save("fruits_learning_transfer_model.h5")

# 7. Test
test_loss, test_acc = model.evaluate(test_data)
st.write(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")  # 🔸 print yerine Streamlit

# 8. Grafikler (Streamlit uyumlu)
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

st.pyplot(plt.gcf())  # 🔸 plt.show yerine bu!
