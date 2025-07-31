import os
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Conv2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

# PARAMETERS
img_size = 128
data_dir = "archive"
epochs = 10

# DATA LOAD FUNCTION
def load_data(image_folder, mask_folder):
    images, masks = [], []
    for file in os.listdir(image_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(image_folder, file)
            mask_path = os.path.join(mask_folder, file.replace(".jpg", ".mat").replace(".png", ".mat"))

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0

            mat_data = scipy.io.loadmat(mask_path)
            if 'groundTruth' in mat_data:
                ground_truth = mat_data['groundTruth']
                if ground_truth.size > 0:
                    mask = ground_truth[0, 0]['Segmentation'][0, 0]
                    mask = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    mask = np.expand_dims((mask > 0).astype(np.float32), axis=-1)
                    images.append(img)
                    masks.append(mask)
    return np.array(images), np.array(masks)

# LOAD DATA
X, y = load_data(
    os.path.join(data_dir, "images", "train"),
    os.path.join(data_dir, "ground_truth", "train")
)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL: RESNET50 + U-NET DECODER
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Skip connections
    c1 = base_model.get_layer("conv1_relu").output
    c2 = base_model.get_layer("conv2_block3_out").output
    c3 = base_model.get_layer("conv3_block4_out").output
    c4 = base_model.get_layer("conv4_block6_out").output
    c5 = base_model.get_layer("conv5_block3_out").output

    # Decoder
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Conv2D(512, 3, activation='relu', padding='same')(u6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(u6)
    u7 = concatenate([u7, c3])
    u7 = Conv2D(256, 3, activation='relu', padding='same')(u7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(u7)
    u8 = concatenate([u8, c2])
    u8 = Conv2D(128, 3, activation='relu', padding='same')(u8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(u8)
    u9 = concatenate([u9, c1])
    u9 = Conv2D(64, 3, activation='relu', padding='same')(u9)

    u10 = Conv2DTranspose(32, 2, strides=2, padding='same')(u9)
    u10 = Conv2D(32, 3, activation='relu', padding='same')(u10)

    outputs = Conv2D(1, 1, activation='sigmoid')(u10)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

model = build_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2)])

# TRAIN
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

# SAVE
model.save("bsds500_resnet50_transfer.h5")

# STREAMLIT UI
st.title("BSDS500 - Transfer Learning (ResNet50) Semantic Segmentation")

st.subheader("Eğitim Sonuçları")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='Train Acc')
ax[0].plot(history.history['val_accuracy'], label='Val Acc')
ax[0].legend(); ax[0].set_title('Accuracy')

ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].legend(); ax[1].set_title('Loss')

st.pyplot(fig)

# PREDICT
st.subheader("Bir Görsel Yükle ve Segmentasyon Yap")
uploaded_file = st.file_uploader("Bir görsel seç...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    prediction = model.predict(img_input)[0, :, :, 0]
    binary_mask = (prediction > 0.5).astype(np.uint8)

    st.image(img_resized, caption="Yüklenen Görsel", width=300)
    st.image(binary_mask * 255, caption="Segmentasyon Maskesi", width=300)
