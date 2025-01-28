# 🧠 Tumor Grade Classification & Segmentation in LGG using Deep Learning

This repository contains a **deep learning-based pipeline** that integrates **U-Net** for **tumor segmentation** and **ResNet** for **tumor grade classification** using MRI images. The project focuses on automating the detection and grading of **Lower-Grade Gliomas (LGG)** to assist in clinical decision-making.

---

## 📌 Features
- **Tumor Segmentation**: U-Net architecture to precisely localize tumor regions in MRI images.
- **Tumor Grade Classification**: ResNet-based model to predict tumor grade (Grade 1 or 2).
- **Medical Data Augmentation**: Techniques like rotation, flipping, resizing, and color jittering to improve performance.
- **Performance Metrics**: Dice Coefficient for segmentation, Accuracy, F1-Score, and ROC-AUC for classification.

---

## 🏥 Problem Statement
- **Lower-Grade Gliomas (LGG)** are slow-growing brain tumors that require precise segmentation and grading for treatment planning.
- **Challenges**:
  - Small dataset size makes deep learning models prone to **underfitting**.
  - **Subtle differences** between normal and tumor tissues complicate segmentation.
  - Generalizing across different MRI scanners is difficult.

This project aims to **automate** the segmentation and classification of LGG tumors to enhance efficiency in **clinical decision-making**.

---

## 🛠 Methodology
### 1️⃣ Dataset: LGG Segmentation Dataset
- Contains **MRI images**, **segmentation masks**, and **clinical metadata** from **110 patients**.
- MRI slices include **FLAIR, pre-contrast, and post-contrast** sequences.

### 2️⃣ Model Architectures
- **U-Net for Segmentation**:
  - Encoder-Decoder architecture with **skip connections**.
  - Loss Function: **Binary Cross-Entropy + Dice Loss**.

- **ResNet50 for Classification**:
  - Pre-trained ResNet50 fine-tuned for tumor grade classification.
  - Loss Function: **Binary Cross-Entropy (BCE)**.

### 3️⃣ Data Augmentation
- **Training**: Applied **random rotation, flipping, contrast adjustments** to prevent overfitting.
- **Testing**: Resized images to 256x256 pixels for consistency.

---

## 📈 Results
### **Segmentation Performance**
- **Dice Coefficient Before Fine-Tuning**: **0.0711**
- **Dice Coefficient After Fine-Tuning**: **0.3854**
- **Observation**: Fine-tuning significantly improved segmentation accuracy.

### **Classification Performance**
- **Accuracy**: **87.63%**
- **F1-Score**: **0.88**
- **AUC (ROC Curve)**: **0.95**
- **Observation**: ResNet achieved strong classification accuracy even with limited fine-tuning.

![ROC Curve](visuals/roc.png)

---

## 🔥 Challenges & Future Improvements
### 🚧 **Challenges Faced**
- **Limited Dataset**: Small sample size restricts model generalization.
- **Long Training Time**: Training took **13+ hours** due to computational constraints.
- **Segmentation Complexity**: Fine-tuning improved results but still needs further optimization.

### 🚀 **Future Work**
1. **Improve Segmentation**: Explore **Transformer-based architectures**.
2. **Synthetic Data Generation**: Use **GANs** for dataset augmentation.
3. **Cross-Dataset Validation**: Test on external MRI datasets for better generalization.

---

## 📜 References
- **U-Net for Medical Segmentation**: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- **ResNet for Image Classification**: [He et al., 2016](https://arxiv.org/abs/1512.03385)
- **LGG Segmentation Dataset**: [Kaggle Source](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
