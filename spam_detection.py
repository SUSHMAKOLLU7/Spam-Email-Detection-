# Import libraries
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("spam.csv")

# Convert labels into numbers
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Separate input and output
x = data['message']
y = data['label']

# Convert text into vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

# Split dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)

# Predict output
predictions = model.predict(x_test)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Test message
msg = ["Congratulations! You won a prize"]

msg_vector = vectorizer.transform(msg)

result = model.predict(msg_vector)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")