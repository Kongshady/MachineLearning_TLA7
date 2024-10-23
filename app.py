from flask import Flask, render_template, request  
import joblib  
import pandas as pd  

app = Flask(__name__)  

# Load the trained model and vectorizer  
model = joblib.load('models/best_model.pkl')  
vectorizer = joblib.load('models/vectorizer.pkl')  

@app.route('/')  
def index():  
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])  
def predict():  
    # Get the text input from the user  
    text = request.form['text']  
    
    # Preprocess the input as done during training  
    text = text.lower()  # Convert to lowercase  
    text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove punctuation  

    # Vectorize the input  
    text_vectorized = vectorizer.transform([text])  

    # Predict the emotion  
    prediction = model.predict(text_vectorized)  
    
    return render_template('index.html', prediction=prediction[0])  

if __name__ == "__main__":  
    app.run(debug=True)