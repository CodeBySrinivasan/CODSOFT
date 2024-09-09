import pickle

# Load the model (which includes the fitted vectorizer) from the file
with open('model/spam_classifier.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Testing the pipeline with a more balanced set of sample data
sample_messages = [
    "Congratulations! You've won a lottery.",  # Spam
    "Let's meet for lunch tomorrow.",           # Ham
    "I'm really not up to it still tonight babe", # Ham
    "Limited time offer! Buy now and save big!", # Spam
    "Can we reschedule our meeting to next week?", # Ham
    "Earn money fast! Click here to find out more.", # Spam
    "Rofl. Its true to its name",               # Ham
    "Free gift card! Click here to claim your prize.", # Spam
    "Looking forward to our catch-up next week.", # Ham
    "Win a new car! No purchase necessary."     # Spam
]

predictions = pipeline.predict(sample_messages)
for message, prediction in zip(sample_messages, predictions):
    print(f"Message: {message}\nPrediction: {prediction}\n")
