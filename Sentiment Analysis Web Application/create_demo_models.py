import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Sample data for demonstration
sample_texts = [
    "I love this product! It's amazing!",
    "This is the worst thing ever.",
    "I'm so happy today!",
    "This is terrible and disappointing.",
    "Great job! Well done!",
    "I hate this so much.",
    "Perfect! Exactly what I needed!",
    "This is awful and broken.",
    "Wonderful experience!",
    "Completely useless."
]

sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

print("Creating demo TF-IDF vectorizer...")
# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=500000,  # 500K features as mentioned in your code
    ngram_range=(1, 2),   # 1-2 grams as mentioned
    stop_words='english'
)

# Fit the vectorizer on sample data
X = vectorizer.fit_transform(sample_texts)

print("Creating demo Logistic Regression model...")
# Create and fit Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X, sample_labels)

print("Saving vectorizer...")
# Save the vectorizer
with open('model/vectoriser-ngram-(1,2).pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Saving model...")
# Save the model
with open('model/Sentiment-LR.pickle', 'wb') as f:
    pickle.dump(model, f)

print("Demo models created successfully!")
print("Files created:")
print("- model/vectoriser-ngram-(1,2).pickle")
print("- model/Sentiment-LR.pickle")

# Test the models
print("\nTesting the models...")
test_text = ["I love this!"]
processed_text = vectorizer.transform(test_text)
prediction = model.predict(processed_text)
probability = model.predict_proba(processed_text)

print(f"Test text: {test_text[0]}")
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {max(probability[0]) * 100:.1f}%")