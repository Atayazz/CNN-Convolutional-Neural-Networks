# 🥦 Fruits & Vegetables Classification with CNN and Transfer Learning

This project implements two deep learning approaches to classify images of fruits and vegetables:

1. **`fruitsandveg_image.py`** – A Convolutional Neural Network (CNN) built from scratch  
2. **`fruits_transfer_learning.py`** – A Transfer Learning model using EfficientNetB0

Both models are deployed using **Streamlit**, allowing for interactive training visualizations and image-based predictions.

---

## 📁 Dataset

**Dataset Used:** [Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

**Directory structure:**

```
archive/
├── train/
├── validation/
└── test/
```

Each subfolder contains one class of fruit or vegetable images.

---

## ⚙️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run CNN model:
```bash
streamlit run fruitsandveg_image.py
```

### Run Transfer Learning model:
```bash
streamlit run fruits_transfer_learning.py
```

---

## 🧠 Model 1: CNN (`fruitsandveg_image.py`)

**Key Features:**
- Custom-built CNN with 5 Conv layers and MaxPooling
- Dense and Dropout for regularization
- Image normalization
- Accuracy and Loss visualizations
- Predicts user-uploaded image
- Saves model as `fruits_veggies_model.h5`

---

## 🧠 Model 2: Transfer Learning (`fruits_transfer_learning.py`)

**Key Features:**
- Based on pre-trained **EfficientNetB0**
- Frozen base layers
- GlobalAveragePooling + Dense head
- Uses ImageDataGenerator for augmentation
- Accuracy and Loss visualizations
- Predicts user-uploaded image
- Saves model as `fruits_transfer_model.h5`

---

## 🖼️ Streamlit Interface Includes:

- Real-time training graphs (Accuracy & Loss)
- Image uploader component
- Prediction with class label shown on screen

---

## 💾 Model Output Files

| Model             | File Name                   | Description              |
|------------------|-----------------------------|--------------------------|
| CNN              | `fruits_veggies_model.h5`    | From-scratch CNN model   |
| Transfer Learning| `fruits_transfer_model.h5`   | EfficientNetB0 model     |

---
