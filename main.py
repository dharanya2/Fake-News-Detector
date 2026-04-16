import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/news.csv")

# Features and labels
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom input
while True:
    news = input("\nEnter news text (or 'exit'): ")
    if news.lower() == "exit":
        break
    
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)
    
    print("Prediction:", prediction[0])