import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Assuming you have your trained model and vectorizer here
# Example initialization (replace these with your actual trained model and vectorizer)
model = MultinomialNB()
vectorizer = TfidfVectorizer()

# Save the model and vectorizer
try:
    with open('model/spam_classifier.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('model/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
except Exception as e:
    print(f"Error occurred while saving the model and vectorizer: {e}")