from flask import Flask, request, jsonify, render_template
import joblib
import boto3
import uuid
import requests

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Configure DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('NewsArticles')

# Configure GDELT API
GDELT_API_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news article from the request
    data = request.form['news_article']

    # Preprocess the input data
    processed_data = vectorizer.transform([data])

    # Predict using the model
    prediction = model.predict(processed_data)

    # Store the result in DynamoDB with a unique ID
    news_id = str(uuid.uuid4())
    table.put_item(Item={'news_id': news_id, 'article': data, 'prediction': int(prediction[0])})

    return jsonify({'prediction': 'Fake' if prediction[0] == 0 else 'Real'})

@app.route('/fetch_news', methods=['GET'])
def fetch_news():
    params = {
        'query': 'fake news',
        'mode': 'ArtList',
        'maxrecords': 10,
        'format': 'json'
    }
    response = requests.get(GDELT_API_URL, params=params)
    data = response.json()

    articles = [{'title': article.get('title', 'No title'), 'description': article.get('seendescription', 'No description')} for article in data.get('articles', [])]
    return jsonify({'articles': articles})

if __name__ == "__main__":
    app.run(debug=True)
