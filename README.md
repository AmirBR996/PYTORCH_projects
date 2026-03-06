# 🚀 PyTorch Deep Learning Projects Portfolio

A comprehensive collection of Deep Learning projects built using PyTorch.  
This repository demonstrates strong knowledge of neural network design, training pipelines, optimization techniques, and real-world AI applications across Computer Vision, NLP, Time Series Forecasting, and Regression tasks.

---

## 📂 Project Structure

PYTORCH_projects/
- Image Classification Projects
  - ANN_fashion_mnist_pytorch.ipynb
  - Fashion_mnist_using_vgg16.ipynb
  - image_classification_digit.ipynb
- Time Series Forecasting
  - Google_Stock_price_prediction.ipynb
  - GOOG.csv
- NLP & Sequence Modeling
  - Next_Word_predictor_using_LSTM.ipynb
  - pytorch_lstm_next_word_predictor.ipynb
  - RNN-QA.ipynb
  - 100_Unique_QA_Dataset.csv
- Regression Projects
  - Car_price_prediction.ipynb
  - Housing_using_ANN.ipynb
  - graduate_Addmission.ipynb
  - Housing.csv
  - movie_review_analysis.ipynb
  - movie.py
- Utility Scripts
  - glove.py
  - movie.py

---

## 🧠 Projects Overview

### 1️⃣ Fashion MNIST Image Classification (ANN)
- Input: 784 features (28x28 images)
- Architecture: 784 → 128 → 64 → 10
- Activation: ReLU
- Loss: CrossEntropyLoss
- Optimizer: SGD
- Accuracy: ~95%

### 2️⃣ VGG16 Transfer Learning
- Pretrained VGG16 (ImageNet)
- Custom classifier head
- Fine-tuning selected layers
- Dropout for regularization

### 3️⃣ Google Stock Price Prediction (LSTM)
- Sequence Length: 60 days
- Model 1: 2-layer LSTM (50 hidden units)
- Model 2: 3-layer LSTM (100 hidden units)
- Loss: MSE
- Optimizer: Adam
- Model 2 improved MSE by ~35%

### 4️⃣ Next Word Prediction (LSTM)
- Dataset: 6,500+ article titles
- Vocabulary: 8,000+ tokens
- Architecture: Embedding → LSTM → Linear → Softmax
- Final Loss: ~2.22
- Example:
  Input: "Introduction to"
  Output: "the attention economy and economic"

### 5️⃣ Question Answering System (RNN)
- Vocabulary size: 324
- Architecture: Embedding → RNN → Linear
- Loss: CrossEntropyLoss
- Example:
  Q: What is the largest planet?
  A: jupiter

---

## 🛠 Tech Stack

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- NLTK
- Jupyter Notebook

---

## ⚙️ Installation

git clone https://github.com/AmirBR996/PYTORCH_projects.git  
cd PYTORCH_projects  
python -m venv venv  
source venv/bin/activate  (Windows: venv\Scripts\activate)  
pip install -r requirements.txt  

---

## 🏋️ Generic Training Loop

for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

---

## 📊 Results Summary

- Fashion MNIST ANN: ~95% Accuracy
- Stock LSTM: 35% MSE improvement
- Next Word LSTM: 53% loss reduction
- QA RNN: 97% loss reduction

---

## 🎯 Key Learning Outcomes

- Data preprocessing & normalization
- Neural network architecture design
- Optimization strategies
- Transfer learning
- Sequence modeling with RNN/LSTM
- Time series forecasting
- GPU training with CUDA

---

## 📌 Future Improvements

- Transformer-based NLP models
- ResNet architectures
- Hyperparameter tuning
- Model deployment using Flask/FastAPI
- Docker & ONNX export

---

## 📬 Contact

GitHub: AmirBR996  

---

## 📜 License

This project is licensed under the MIT License.
