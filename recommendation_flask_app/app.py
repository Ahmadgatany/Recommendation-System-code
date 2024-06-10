# app.py
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('kmeans_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    longitude = float(request.form['longitude'])
    latitude = float(request.form['latitude'])
    
    # Perform prediction
    label = model.predict([[longitude, latitude]])[0]
    
    return render_template('index.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
