import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gensim.downloader as api
import re

nltk.download("punkt")

df = pd.read_csv("IMDBDataset.csv")
X = df['review']
Y = df['sentiment'].map({"negative": 0, "positive": 1})

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)
    return tokens

X_train = X.apply(clean_text)
X_test = X.apply(clean_text)

# Load Word2Vec model
word2vec = api.load("word2vec-google-news-300")
embedding_dim = 300

# Convert text to embeddings
def get_embedding(tokens, embedding_dim=300):
    vectors = [word2vec[word] for word in tokens if word in word2vec]
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)

X_train = X_train.apply(lambda x: get_embedding(x))
X_test = X_test.apply(lambda x: get_embedding(x))

# Dataset Class
class ReviewData(Dataset):
    def __init__(self, X_data, Y_data):
        self.reviews = torch.tensor(np.array(X_data.tolist()), dtype=torch.float32)
        self.labels = torch.tensor(Y_data.tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        return self.reviews[item], self.labels[item]

train_dataset = ReviewData(X_train, Y)
test_dataset = ReviewData(X_test, Y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Model
class Sentiment_model(nn.Module):
    def __init__(self, input_dim):
        super(Sentiment_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = F.relu(self.bn1(self.fc1(X)))
        X = self.dropout(X)
        X = F.relu(self.bn2(self.fc2(X)))
        X = self.dropout(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return self.sigmoid(X)

model = Sentiment_model(embedding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0

    model.train()
    for reviews, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(reviews)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = (correct_train / total_train) * 100

    # Evaluation
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(reviews)
            predicted = (outputs > 0.5).float()
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_accuracy = (correct_test / total_test) * 100

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

# Prediction Function
def predict_sentiment(model, review_text):
    tokens = clean_text(review_text)
    review_vector = get_embedding(tokens)
    review_tensor = torch.tensor(review_vector, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(review_tensor)

    sentiment = "Positive" if output.item() > 0.5 else "Negative"
    return sentiment, output.item()

sample_review = "An emotional rollercoaster that left me in tears. Absolutely loved it!"
sentiment, confidence = predict_sentiment(model, sample_review)
print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.4f})")
