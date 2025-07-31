import os
import numpy as np
import cv2
import scipy.io
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split

# 1. Parameters
img_size = 128
data_dir = "archive"
epochs = 10

# 2. Load images and masks
def load_data(image_folder, mask_folder):
    images = []
    masks = []

    image_files = os.listdir(image_folder)

    for file in image_files:
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(image_folder, file)
            mask_path = os.path.join(mask_folder, file.replace(".jpg", ".mat").replace(".png", ".mat"))

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # normalize image

            mat_data = scipy.io.loadmat(mask_path)
            if 'groundTruth' in mat_data:
                ground_truth = mat_data['groundTruth']
                if ground_truth.size > 0:
                    mask = ground_truth[0, 0]['Segmentation'][0, 0]
                    mask = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    mask = np.expand_dims(mask, axis=-1)
                    mask = (mask > 0).astype(np.float32)  # Ensure binary mask
                    images.append(img)
                    masks.append(mask)

    return np.array(images), np.array(masks)

# Load data
subset = "train"
X, y = load_data(os.path.join(data_dir, "images", subset), os.path.join(data_dir, "ground_truth", subset))

if X.size == 0 or y.size == 0:
    raise ValueError("No images or masks were loaded. Check the directory structure and file formats.")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. U-Net Model
def unet_model():
    inputs = Input((img_size, img_size, 3))

    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, 3, activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(256, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(128, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(64, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(32, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2)])

# 4. Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

# 5. Save the model
model.save("bsds500_cnn_model.h5")

# 6. Streamlit UI
st.title("BSDS500 - U-Net Image Segmentation")
st.subheader("Training Accuracy & Loss")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[0].legend()
ax[0].set_title('Accuracy')

ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Val Loss')
ax[1].legend()
ax[1].set_title('Loss')

st.pyplot(fig)

# 7. Prediction Section
st.subheader("Upload an Image to Segment")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    prediction = model.predict(img_input)[0, :, :, 0]
    binary_mask = (prediction > 0.5).astype(np.uint8)

    # Show images
    st.image(img_resized, channels="BGR", caption="Uploaded Image", width=300)
    st.image(binary_mask * 255, caption="Predicted Segmentation Mask", width=300)
