# 🫁 Pneumonia Detection using DenseNet121

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project aims to build a robust Deep Learning model to classify Chest X-Ray images into two categories: **Normal** and **Pneumonia**. 

In medical diagnosis, **Recall (Sensitivity)** is a critical metric because missing a positive case (False Negative) is far more dangerous than a False Positive. This project leverages **Transfer Learning** with the **DenseNet121** architecture to achieve high sensitivity in detecting pneumonia cases.

## 📂 Dataset
The dataset used in this project is the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.
- **Structure:** The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal).
- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## ⚙️ Methodology

### 1. Data Preprocessing & Augmentation
To handle class imbalance and prevent overfitting, the following techniques were applied:
- **Rescaling:** Pixel values normalized to [0,1].
- **Augmentation:** Rotation (20°), Zoom (0.2), Width/Height Shifts, and Horizontal Flips applied to training data.

### 2. Model Architecture: DenseNet121
We utilized **Transfer Learning** using the **DenseNet121** model pre-trained on ImageNet.
- **Base Model:** DenseNet121 (Frozen weights).
- **Custom Head:** Global Average Pooling -> Batch Normalization -> Dropout (0.5) -> Dense (128, ReLU) -> Dropout (0.3) -> Output (Sigmoid).
- **Optimizer:** Adam (Learning Rate: 0.001) with `ReduceLROnPlateau`.

## 📊 Results
The model was evaluated on the test set (624 images).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **91%** |
| **Pneumonia Recall** | **93%** |
| **Pneumonia Precision** | **92%** |
| **F1-Score** | **93%** |

**Classification Report:**
```text
              precision    recall  f1-score   support

      Normal       0.89      0.86      0.87       234
   Pneumonia       0.92      0.93      0.93       390

    accuracy                           0.91       624
   macro avg       0.90      0.90      0.90       624
weighted avg       0.91      0.91      0.91       624
