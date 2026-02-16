# Brain Tumor Detection using Deep Learning (VGG16 Transfer Learning)

## Project Overview

This project implements a deep learning–based system for automated brain tumor classification using MRI scans. The model leverages transfer learning with the VGG16 architecture to classify brain MRI images into four clinically relevant categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The project also includes a Flask-based web application that allows real-time tumor prediction by uploading MRI images.

This system demonstrates a complete end-to-end deep learning pipeline including:

- Dataset preprocessing
- Transfer learning with VGG16
- Model training and validation
- Performance evaluation
- Model saving and loading
- Deployment via web application

---

## Dataset

The dataset used in this project is publicly available on Kaggle:

Brain MRI Images for Brain Tumor Detection  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download the dataset and place it inside the following structure:

dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/


## Trained Model

The trained VGG16 model is available on Hugging Face:

https://huggingface.co/muhammedpanchla/brain-tumor-detection-vgg16/tree/main

Download the model and place it inside:

model/
│
└── brain_tumor_cnn.pth


## Model Architecture

Base model: VGG16 (pretrained on ImageNet)

Transfer learning approach:

- Feature extraction using pretrained convolutional layers
- Custom fully connected classifier added
- Fine-tuned final layers for MRI classification task

Technical details:

- Framework: PyTorch
- Input size: 224 × 224 pixels
- Number of classes: 4
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Training epochs: 10

---

## Model Performance

Final performance metrics:

- Training Accuracy: 99.35%
- Validation Accuracy: 97.14%

The model demonstrates excellent generalization performance and strong capability in detecting and classifying brain tumors from MRI scans.

Training progression highlights:

- Initial validation accuracy: 90.71%
- Final validation accuracy: 97.14%
- Best model saved based on validation performance


---

## Project Structure

brain-tumor-detection-cnn/
│
├── dataset/
├── model/
├── notebooks/
├── app/
├── results/
├── README.md
├── requirements.txt


---

## Training Performance

The model was trained for 10 epochs using transfer learning with VGG16.

Training progression:

Epoch 1:
- Training Accuracy: 84.06%
- Validation Accuracy: 90.71%

Epoch 5:
- Training Accuracy: 98.79%
- Validation Accuracy: 96.61%

Final Epoch:
- Training Accuracy: 99.35%
- Validation Accuracy: 97.14%

The best model was saved based on validation accuracy to ensure optimal generalization.


## Results

Training Accuracy and Loss curves:

(results/training_accuracy.png)

(results/training_loss.png)

Sample Predictions:

(results/sample_prediction.png)

---

## Web Application

A Flask-based web application allows users to upload MRI images and receive tumor predictions.

Run locally using:

cd app
python app.py


---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Flask
- NumPy
- OpenCV
- Matplotlib

---

## Author

Mohammed Panchla

Aspiring Machine Learning Engineer specializing in Medical Imaging AI.
