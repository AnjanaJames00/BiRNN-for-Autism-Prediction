# 🧠 BiRNN for Autism Prediction

This project implements a **Bidirectional Recurrent Neural Network (BiRNN)** using Bidirectional LSTM layers for autism-related prediction and classification tasks from speech-based features.

The model processes audio recordings, extracts MFCC features, and learns temporal patterns using deep recurrent neural networks to support autism prediction research.

---

## 📌 Overview

Autism Spectrum Disorder (ASD) affects communication and social interaction patterns. Speech signals often contain useful acoustic characteristics that can be analyzed using machine learning and deep learning techniques.

This project uses:

- Audio signal processing
- MFCC feature extraction
- Bidirectional LSTM networks
- Deep learning classification

to learn meaningful patterns from speech recordings.

---

## 🚀 Features

- Automated dataset download using KaggleHub
- Audio preprocessing with Librosa
- MFCC feature extraction
- Feature normalization
- Deep Bidirectional LSTM architecture
- Early stopping for improved generalization
- Training and validation performance visualization

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-Learn
- Matplotlib
- KaggleHub

---

## 📂 Dataset

Dataset downloaded from Kaggle:

https://www.kaggle.com/datasets/preethikurra/ravdess-tess

The dataset contains speech recordings that are processed to extract acoustic features for model training.

---

## 🔍 Feature Extraction

### MFCC Features

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio sample.

Processing pipeline:

1. Load audio file
2. Extract 40 MFCC coefficients
3. Pad/truncate sequences to fixed length
4. Normalize features
5. Convert to sequence format for BiRNN

Input shape:

```python
(samples, 100, 40)
```

---

## 🧠 Model Architecture

```text
Input Layer
(100 × 40)

↓
Bidirectional LSTM (128)

↓
Dropout (0.3)

↓
Bidirectional LSTM (128)

↓
Dropout (0.3)

↓
Bidirectional LSTM (64)

↓
Dropout (0.3)

↓
Dense (64, ReLU)

↓
Output Layer (Softmax)
```

---

## ⚙️ Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 📈 Training Configuration

| Parameter | Value |
|------------|---------|
| Epochs | 40 |
| Batch Size | 32 |
| Optimizer | Adam |
| Early Stopping | Yes |
| Patience | 5 |

---

## 📊 Results

The model tracks:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

Performance graphs are generated after training.

---
