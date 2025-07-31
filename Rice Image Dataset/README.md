# 🌾 Rice Classification with CNN and Transfer Learning

This project classifies different types of rice grains using two deep learning approaches:

- `rice_image.py`: A custom CNN model built from scratch (no transfer learning)
- `rice_transfer.py`: A model using **Transfer Learning** with **ResNet50**

The application uses **Streamlit** for easy interactive deployment.

---

## 📦 Requirements

Install the required dependencies using:

```
pip install -r requirements.txt
```

Or manually install:

```
pip install tensorflow numpy opencv-python scikit-learn matplotlib streamlit
```

---

## 1️⃣ rice_image.py - CNN from Scratch

This script builds a deep Convolutional Neural Network without using any pre-trained models.

### ✅ Features

- 5 convolutional layers and 5 pooling layers
- Dropout for regularization
- Data normalization and label encoding
- Metrics: `accuracy`, `precision`, `recall`
- Saves model as `rice_model.h5`
- Streamlit app for image upload and prediction
- Accuracy & loss training plots

### ▶️ Run the app

```
streamlit run rice_image.py
```

---

## 2️⃣ rice_transfer.py - Transfer Learning with ResNet50

This script uses ResNet50 with pre-trained ImageNet weights and a custom classification head.

### ✅ Features

- Pretrained ResNet50 model (`include_top=False`)
- Custom classification head with Dense + Dropout layers
- Freezing base layers for feature extraction
- Data augmentation with `ImageDataGenerator`
- Metrics: `accuracy`, `precision`, `recall`
- Saves model as `rice_resnet_model.h5`
- Streamlit app for image upload and prediction
- Accuracy & loss training plots

### ▶️ Run the app

```
streamlit run rice_transfer.py
```

---

## 🧪 Example Output

After training, both apps provide:

- Training accuracy & loss plots
- File uploader to classify new rice images
- Predicted class shown alongside the uploaded image

---

## 📚 Dataset Information

**Dataset:** [Rice Image Dataset](https://www.kaggle.com/datasets/rahulraut29/rice-image-dataset)

**Classes:**

- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

Each class contains 200 images, totaling 1000 labeled rice grain images.

---

## 💾 Model Files

| Model              | File Name              | Type        |
|-------------------|------------------------|-------------|
| CNN from Scratch  | `rice_model.h5`        | Keras model |
| ResNet50 Model     | `rice_resnet_model.h5` | Keras model |
| Label Encoder     | `label_encoder.pkl`    | Pickle file |

---

