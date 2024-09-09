import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Sample training data (update this with your actual data)
training_messages = [
    "Free money!!!", 
    "Hello, how are you?", 
    "Win big prizes", 
    "Call me now", 
    "Meeting at noon"
]
training_labels = ['spam', 'ham', 'spam', 'ham', 'ham']

# Print class distribution
print("Class distribution:", Counter(training_labels))

# Initialize vectorizer and model
vectorizer = TfidfVectorizer()
model = MultinomialNB()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(training_messages, training_labels, test_size=0.2, random_state=42)

# Create a pipeline and fit it
pipeline = make_pipeline(vectorizer, model)
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Create the 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the fitted pipeline to a file
try:
    with open('model/spam_classifier.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("Model and vectorizer saved successfully.")
except Exception as e:
    print(f"Error occurred while saving the model and vectorizer: {e}")
