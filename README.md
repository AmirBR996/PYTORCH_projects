🚀 PyTorch Deep Learning Projects Portfolio

A comprehensive collection of Deep Learning projects built using PyTorch.
This repository demonstrates strong understanding of:

Neural Network Architecture Design

Model Training & Optimization

NLP & Sequence Modeling

Time Series Forecasting

Transfer Learning

End-to-End ML Pipelines

📂 Project Structure
PYTORCH_projects/
│
├── Image Classification Projects/
│   ├── ANN_fashion_mnist_pytorch.ipynb
│   ├── Fashion_mnist_using_vgg16.ipynb
│   └── image_classification_digit.ipynb
│
├── Time Series Forecasting/
│   ├── Google_Stock_price_prediction.ipynb
│   └── GOOG.csv
│
├── NLP & Sequence Modeling/
│   ├── Next_Word_predictor_using_LSTM.ipynb
│   ├── pytorch_lstm_next_word_predictor.ipynb
│   ├── RNN-QA.ipynb
│   └── 100_Unique_QA_Dataset.csv
│
├── Regression Projects/
│   ├── Car_price_prediction.ipynb
│   ├── Housing_using_ANN.ipynb
│   ├── graduate_Addmission.ipynb
│   ├── Housing.csv
│   ├── movie_review_analysis.ipynb
│   └── movie.py
│
└── Utility Scripts/
    ├── glove.py
    └── movie.py
🧠 Projects
1️⃣ Fashion MNIST Image Classification (ANN)

Problem: Classify clothing images into 10 categories.

Model Architecture
Input (784)
→ Linear (128) + ReLU
→ Linear (64) + ReLU
→ Linear (10)

Loss Function: CrossEntropyLoss

Optimizer: SGD

Epochs: 100

Accuracy: ~95%

2️⃣ VGG16 Transfer Learning

Goal: Improve classification performance using pre-trained CNN.

Pretrained VGG16 (ImageNet)

Custom classifier head

Dropout for regularization

Fine-tuning selected layers

3️⃣ Google Stock Price Prediction (LSTM)

Problem: Predict stock closing prices using historical sequences.

Model 1

LSTM (Hidden: 50, Layers: 2)

Model 2

LSTM (Hidden: 100, Layers: 3)

Sequence Length: 60 days

Loss: MSE

Optimizer: Adam

Model 2 improved MSE by ~35%

4️⃣ Next Word Prediction (LSTM)

Goal: Generate contextual next words.

Architecture
Embedding Layer
→ LSTM
→ Fully Connected Layer
→ Softmax Output

Dataset: 6,500+ article titles

Vocabulary: 8,000+ tokens

Final Loss: ~2.22

Example:

Input: "Introduction to"
Output: "the attention economy and economic"
5️⃣ Question-Answering System (RNN)

Goal: Predict correct answers for given questions.

Embedding Layer

RNN (Hidden: 64)

Linear Output Layer

Vocabulary Size: 324

Example:

Q: What is the largest planet?
A: jupiter
🛠 Tech Stack

Python 3.7+

PyTorch

NumPy

Pandas

Scikit-learn

Matplotlib

NLTK

Jupyter Notebook

⚙️ Installation
git clone https://github.com/AmirBR996/PYTORCH_projects.git
cd PYTORCH_projects

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
🏋️ Training Example (Generic)
for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
📊 Results Summary
Project	Performance
Fashion MNIST (ANN)	~95% Accuracy
Stock Prediction (LSTM)	35% MSE improvement
Next Word LSTM	53% loss reduction
QA System	97% loss reduction
🔮 Future Improvements

Transformer models for NLP

ResNet for image classification

Hyperparameter tuning

Model deployment using Flask / FastAPI

ONNX export

Docker containerization

📌 Learning Outcomes

Data preprocessing techniques

Neural network design

Loss functions & optimizers

Transfer learning

Sequence modeling

GPU training with CUDA

📬 Contact

GitHub: AmirBR996

📜 License

This project is licensed under the MIT License.
