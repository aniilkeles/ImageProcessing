# Waste Classification using Deep Learning (MobileNetV2)

This project is a real-time **Waste Classification System** developed using **TensorFlow**, **Keras**, and **OpenCV**. The model classifies waste into 5 categories: **Cardboard, Glass, Metal, Paper, and Plastic**.

## 🚀 Project Overview
The goal of this project is to automate waste sorting to improve recycling efficiency. I used **Transfer Learning** with the **MobileNetV2** architecture to achieve high accuracy even with a limited dataset.

## 📊 Performance & Results
The model was trained for 15 epochs and achieved the following results:
- **Training Accuracy:** ~93%
- **Validation Accuracy:** ~81%
- **Base Model:** MobileNetV2 (Pre-trained on ImageNet)

### Confusion Matrix Insights
Based on the evaluation phase:
- **Best Performing Classes:** Paper and Metal (nearly 98% accuracy).
- **Observation:** Some plastics were misclassified as glass due to similar transparency and reflective properties, which is a common challenge in computer vision for materials.

## 🛠️ Tech Stack
- **Framework:** TensorFlow & Keras
- **Image Processing:** OpenCV
- **Data Analysis:** Matplotlib, Seaborn, NumPy
- **Dataset Management:** Python ImageDataGenerator (Data Augmentation)

## 📁 File Structure
- `main.py`: Script for training the model.
- `predict.py`: Real-time classification via webcam.
- `evaluate.py`: Performance analysis and Confusion Matrix generation.
- `.gitignore`: Configured to exclude large dataset files and model weights for clean repository management.

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow opencv-python matplotlib seaborn
