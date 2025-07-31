# 🍇 Grape Disease Classification with CNN and Transfer Learning

This project implements two deep learning models to classify grape leaf diseases:

1. **`grapevine_cnn.py`** – A Convolutional Neural Network (CNN) built from scratch  
2. **`grape_transfer_resnet.py`** – A Transfer Learning model using **ResNet50**

Both models are deployed using **Streamlit**, allowing for training visualization and image-based prediction.

---

## 📁 Dataset

**Dataset Used:** Augmented Grape Disease Dataset  
**Folder Name:** `Final Training Data`

Each subfolder contains images labeled for a specific grape disease category:
- Black Rot
- ESCA
- Leaf Blight
- Healthy

---

## ⚙️ How to Run

Install required packages:

```bash
pip install -r requirements.txt
```

### Run CNN model:
```bash
streamlit run grapevine_cnn.py
```

### Run Transfer Learning model:
```bash
streamlit run grape_transfer_resnet.py
```

---

## 🧠 Model 1: CNN (`grapevine_cnn.py`)

**Key Features:**
- 5 Convolutional layers + 3 MaxPooling layers
- Dropout layers for regularization
- Precision, Recall, and Accuracy metrics
- Uses `train_test_split` for dataset split
- Accuracy and Loss visualizations with Matplotlib
- Streamlit interface with image upload for prediction
- Model saved as `grape_cnn_model.h5`

---

## 🧠 Model 2: Transfer Learning (`grape_transfer_resnet.py`)

**Key Features:**
- Based on pre-trained **ResNet50** (ImageNet weights)
- Frozen convolutional base layers
- Classification head with `GlobalAveragePooling2D` + Dense layers
- Uses `ImageDataGenerator` with augmentation and validation split
- EarlyStopping to prevent overfitting
- Accuracy and Loss training curves
- Streamlit image upload and prediction interface
- Model saved as `grape_resnet_model.h5`

---

## 🖼️ Streamlit Interface Includes:

- Real-time training graphs: Accuracy & Loss
- Upload your own grape leaf image for prediction
- Displays predicted disease class with confidence

---

## 💾 Model Output Files

| Model             | File Name               | Description                |
|------------------|-------------------------|----------------------------|
| CNN              | `grape_cnn_model.h5`    | Custom CNN from scratch    |
| Transfer Learning| `grape_resnet_model.h5` | ResNet50-based classifier  |
