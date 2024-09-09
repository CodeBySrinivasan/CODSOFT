from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model (which includes the fitted vectorizer) from the file
with open('model/spam_classifier.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        try:
            # Predict using the loaded pipeline
            prediction = pipeline.predict(data)[0]

            # Print the prediction for debugging purposes
            print(f"Prediction for message '{message}': {prediction}")

            if prediction == 'spam':
                result = "This is a spam message."
            else:
                result = "This is a ham message."
        except Exception as e:
            result = f"Error: {str(e)}"
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
