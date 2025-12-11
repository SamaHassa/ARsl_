1. Overview 🧐

ARsl_Project is an Arabic Sign Language (ArASL) recognition system.
It identifies Arabic letters from hand gesture images using Deep Learning with a pretrained EfficientNetB0 CNN model.
The project is deployed using Streamlit, allowing users to interact easily and get real-time predictions.

2. Features ⚡

✅ Recognizes 32 Arabic sign language letters

✅ Supports image uploads for letter prediction

✅ Provides high accuracy predictions with confidence scores

✅ Visualizes training and validation performance

3. Dataset 📊

Dataset Name: Arabic Alphabets Sign Language Dataset (ARASL)

Dataset Source: Kaggle

Total Images: 54,049

Number of Classes: 32

Data Split

🏋️ Training set: 37,835 images

🧪 Validation set: 16,214 images

📏 Test set: 50% of validation set

4. Model Architecture 🏗️

Base Model: EfficientNetB0 (pretrained on ImageNet)

Input Shape: 224x224x3

Added Layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense (32 classes, softmax)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

5. Training 🚀
Strategy

Initial Training: 10 epochs with frozen base model

Fine-tuning: 5 epochs with all layers trainable

Results

Train Accuracy: 96% ✅

Test Accuracy: 98% 🎯

Notes:

The model converged quickly, achieving high validation accuracy by epoch 5.

6. Deployment 🌐

The project is deployed using Streamlit.

python -m streamlit run app.py


Users can upload images of hand gestures and get the predicted Arabic letter instantly.

7. Dependencies 🧰

Python 3.9+ 🐍

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

8. Dataset Download Example ⬇️
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

9. Model Saving 💾
model.save(r"C:\ARsl_\arsl_efficientnetb0_model.keras")


1. Overview 🧐

ARsl_Project is an Arabic Sign Language (ArASL) recognition system.
It identifies Arabic letters from hand gesture images using Deep Learning with a pretrained EfficientNetB0 CNN model.
The project is deployed using Streamlit, allowing users to interact easily and get real-time predictions.

2. Features ⚡

✅ Recognizes 32 Arabic sign language letters

✅ Supports image uploads for letter prediction

✅ Provides high accuracy predictions with confidence scores

✅ Visualizes training and validation performance

3. Dataset 📊

Dataset Name: Arabic Alphabets Sign Language Dataset (ARASL)

Dataset Source: Kaggle

Total Images: 54,049

Number of Classes: 32

Data Split

🏋️ Training set: 37,835 images

🧪 Validation set: 16,214 images

📏 Test set: 50% of validation set

4. Model Architecture 🏗️

Base Model: EfficientNetB0 (pretrained on ImageNet)

Input Shape: 224x224x3

Added Layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense (32 classes, softmax)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

5. Training 🚀
Strategy

Initial Training: 10 epochs with frozen base model

Fine-tuning: 5 epochs with all layers trainable

Results

Train Accuracy: 96% ✅

Test Accuracy: 98% 🎯

Notes:

The model converged quickly, achieving high validation accuracy by epoch 5.

6. Deployment 🌐

The project is deployed using Streamlit.

python -m streamlit run app.py


Users can upload images of hand gestures and get the predicted Arabic letter instantly.

7. Dependencies 🧰

Python 3.9+ 🐍

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

8. Dataset Download Example ⬇️
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

9. Model Saving 💾
model.save(r"C:\ARsl_\arsl_efficientnetb0_model.keras")

1. Overview 🧐

ARsl_Project is an Arabic Sign Language (ArASL) recognition system.
It identifies Arabic letters from hand gesture images using Deep Learning with a pretrained EfficientNetB0 CNN model.
The project is deployed using Streamlit, allowing users to interact easily and get real-time predictions.

2. Features ⚡

✅ Recognizes 32 Arabic sign language letters

✅ Supports image uploads for letter prediction

✅ Provides high accuracy predictions with confidence scores

✅ Visualizes training and validation performance

3. Dataset 📊

Dataset Name: Arabic Alphabets Sign Language Dataset (ARASL)

Dataset Source: Kaggle

Total Images: 54,049

Number of Classes: 32

Data Split

🏋️ Training set: 37,835 images

🧪 Validation set: 16,214 images

📏 Test set: 50% of validation set

4. Model Architecture 🏗️

Base Model: EfficientNetB0 (pretrained on ImageNet)

Input Shape: 224x224x3

Added Layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense (32 classes, softmax)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

5. Training 🚀
Strategy

Initial Training: 10 epochs with frozen base model

Fine-tuning: 5 epochs with all layers trainable

Results

Train Accuracy: 96% ✅

Test Accuracy: 98% 🎯

Notes:

The model converged quickly, achieving high validation accuracy by epoch 5.

6. Deployment 🌐

The project is deployed using Streamlit.

python -m streamlit run app.py


Users can upload images of hand gestures and get the predicted Arabic letter instantly.

7. Dependencies 🧰

Python 3.9+ 🐍

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

8. Dataset Download Example ⬇️
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

9. Model Saving 💾
model.save(r"C:\ARsl_\arsl_efficientnetb0_model.keras")
1. Overview 🧐

ARsl_Project is an Arabic Sign Language (ArASL) recognition system.
It identifies Arabic letters from hand gesture images using Deep Learning with a pretrained EfficientNetB0 CNN model.
The project is deployed using Streamlit, allowing users to interact easily and get real-time predictions.

2. Features ⚡

✅ Recognizes 32 Arabic sign language letters

✅ Supports image uploads for letter prediction

✅ Provides high accuracy predictions with confidence scores

✅ Visualizes training and validation performance

3. Dataset 📊

Dataset Name: Arabic Alphabets Sign Language Dataset (ARASL)

Dataset Source: Kaggle

Total Images: 54,049

Number of Classes: 32

Data Split

🏋️ Training set: 37,835 images

🧪 Validation set: 16,214 images

📏 Test set: 50% of validation set

4. Model Architecture 🏗️

Base Model: EfficientNetB0 (pretrained on ImageNet)

Input Shape: 224x224x3

Added Layers:

GlobalAveragePooling2D

Dropout (0.3)

Dense (32 classes, softmax)

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

5. Training 🚀
Strategy

Initial Training: 10 epochs with frozen base model

Fine-tuning: 5 epochs with all layers trainable

Results

Train Accuracy: 96% ✅

Test Accuracy: 98% 🎯

Notes:

The model converged quickly, achieving high validation accuracy by epoch 5.

6. Deployment 🌐

The project is deployed using Streamlit.

python -m streamlit run app.py


Users can upload images of hand gestures and get the predicted Arabic letter instantly.

7. Dependencies 🧰

Python 3.9+ 🐍

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

8. Dataset Download Example ⬇️
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

9. Model Saving 💾
model.save(r"C:\ARsl_\arsl_efficientnetb0_model.keras")


![184dd228-0a1d-41dd-85dd-19a9d0c31b42](https://github.com/user-attachments/assets/0d3154e3-7ebd-4bb8-85e1-e90a2e25b9ad)

