# 🧠 BSDS500 Image Segmentation with CNN and Transfer Learning

This project implements two deep learning approaches to perform image segmentation on the **Berkeley Segmentation Dataset (BSDS500)**:

1. **`bsds500_segmentation_cnn.py`** – A U-Net-based Convolutional Neural Network (CNN) built from scratch  
2. **`bsds500_transfer_learning.py`** – A Transfer Learning model using ResNet50 as encoder with a custom decoder

Both models are deployed using **Streamlit**, enabling interactive training visualizations and real-time segmentation predictions.

---

## 📁 Dataset

**Dataset Used:** [BSDS500 - Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

The dataset must be structured as follows:

- `archive/images/train/` → input images (`.jpg` or `.png`)  
- `archive/ground_truth/train/` → corresponding ground truth masks (`.mat` files with `groundTruth` segmentation data)

Each `.jpg` or `.png` file must have a `.mat` file with the same base name.

---

## ⚙️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run CNN model:
```bash
streamlit run bsds500_segmentation_cnn.py
```

### Run Transfer Learning model:
```bash
streamlit run bsds500_transfer_learning.py
```

---

## 🧠 Model 1: CNN (`bsds500_segmentation_cnn.py`)

**Key Features:**
- Custom U-Net with 5+ convolutional layers
- 3+ max pooling layers
- Dropout regularization
- Binary mask prediction via sigmoid output
- Training and validation accuracy/loss plots
- Real-time segmentation of uploaded images
- Model saved as `bsds500_cnn_model.h5`

---

## 🧠 Model 2: Transfer Learning (`bsds500_transfer_learning.py`)

**Key Features:**
- Encoder: Pretrained **ResNet50** (`imagenet` weights)
- Decoder: Custom CNN-based U-Net-like layers
- Skip connections for semantic segmentation
- Binary mask prediction via sigmoid output
- Training and validation metric visualization
- Real-time mask prediction for uploaded images
- Model saved as `bsds500_resnet50_transfer.h5`

---

## 🖼️ Streamlit Interface Includes:

- Training graphs (Accuracy & Loss)
- File uploader to select image
- Predicted segmentation mask display

---

## 💾 Model Output Files

| Model             | File Name                       | Description                         |
|------------------|----------------------------------|-------------------------------------|
| CNN              | `bsds500_cnn_model.h5`           | Scratch-built U-Net segmentation    |
| Transfer Learning| `bsds500_resnet50_transfer.h5`   | ResNet50-based transfer segmentation |
