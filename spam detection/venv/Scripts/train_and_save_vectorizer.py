import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Example training data
messages = ["Congratulations! You've won a lottery.", "Let's meet for lunch tomorrow.", "Earn money fast! Click here to find out more.", "Can we reschedule our meeting to next week?", "I got a call from a landline number. . . I am asked to come to anna nagar . . . I will go in the afternoon"]
labels = ['spam', 'ham', 'spam', 'ham', 'ham']

# Split data
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=2)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
with open('model/spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
