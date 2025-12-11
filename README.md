# ARsl_
Overview

ARsl_Project is an Arabic Sign Language (ArASL) recognition system that identifies Arabic letters from hand gesture images. The system uses Deep Learning with a pretrained EfficientNetB0 CNN model and is deployed using Streamlit for easy user interaction.

Features

Recognizes 32 Arabic sign language letters.

Supports image uploads for letter prediction.

Provides high accuracy predictions with confidence scores.

Visualizes training and validation performance.

Dataset

Dataset: Arabic Alphabets Sign Language Dataset (ARASL)

Total Images: 54,049

Number of Classes: 32

Data Split:

Training: 37,835 images

Validation: 16,214 images

Test: 50% of validation set

Model Architecture

Base Model: EfficientNetB0 (pretrained on ImageNet)

Input Shape: 224x224x3

Added Layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense (32 classes, softmax)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Training

Initial Training: 10 epochs with frozen base model

Fine-tuning: 5 epochs with all layers trainable

Training Results:

Train Accuracy: 96%

Test Accuracy: 98%

Training Curve:

The model converged quickly, achieving high validation accuracy by epoch 5.

Deployment

The project is deployed using Streamlit:

python -m streamlit run app.py


Users can upload images of hand gestures and get the predicted Arabic letter instantly.

Dependencies

Python 3.9+

TensorFlow 2.x

Keras

NumPy

Matplotlib

Seaborn

Pillow (PIL)

scikit-learn

Librosa

Streamlit

Kaggle API

Dataset Download Example
import os
os.makedirs('/root/.kaggle', exist_ok=True)
!mv /content/kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    'gannayasser/arabic-alphabets-sign-language-dataset-arasl',
    path='.',
    unzip=True
)

Model Saving
model.save(r"C:\ARsl_\arsl_efficientnetb0_model.keras")
