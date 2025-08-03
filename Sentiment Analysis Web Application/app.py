from flask import Flask, render_template, request, jsonify
import re
import pickle
import numpy as np
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

app = Flask(__name__)

# Global variables for models
vectoriser = None
model = None

# Emojis dictionary (from your notebook)
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
    ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
    ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\\\': 'annoyed',
    ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
    '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)" : 'sadsmile', ';)': 'wink',
    ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

# Stopwords list (from your notebook)
stopwordlist = [
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and', 'any', 'are',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
    'itself', 'just', 'll', 'm', 'ma', 'me', 'more', 'most', 'my', 'myself', 'now', 'o',
    'of', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own',
    're', 's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such', 't',
    'than', 'that', "thatll", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
    'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
    "youve", 'your', 'yours', 'yourself', 'yourselves'
]

def preprocess(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()

    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = r'@[^\s]+'
    alphaPattern = r"[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern, ' URL', tweet)

        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

        tweet = re.sub(userPattern, ' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')

        processedText.append(tweetwords)

    return processedText

def load_models():
    global vectoriser, model

    try:
        with open('model/vectoriser-ngram-(1,2).pickle', 'rb') as file:
            vectoriser = pickle.load(file)

        with open('model/Sentiment-LR.pickle', 'rb') as file:
            model = pickle.load(file)

        print("Models loaded successfully!")
        print("- Vectorizer: vectoriser-ngram-(1,2).pickle")
        print("- Model: Sentiment-LR.pickle (Logistic Regression)")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        try:
            for file in os.listdir('model'):
                print(f"  - {file}")
        except:
            print("  Could not list model directory contents")
        return False

def predict_sentiment(text_list):
    if vectoriser is None or model is None:
        return None

    processed_text = preprocess(text_list)
    text_vector = vectoriser.transform(processed_text)
    predictions = model.predict(text_vector)
    probabilities = model.predict_proba(text_vector)

    results = []
    for i, (text, pred) in enumerate(zip(text_list, predictions)):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(probabilities[i]) * 100

        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 2)
        })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict_page.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        results = predict_sentiment([text])

        if results is None:
            return jsonify({'error': 'Models not loaded'}), 500

        return jsonify({
            'success': True,
            'result': results[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance')
def performance():
    metrics = {
        'models_trained': ['Logistic Regression', 'Bernoulli Naive Bayes', 'Linear SVC'],
        'best_model': 'Logistic Regression',
        'dataset_size': '1.6M tweets',
        'features': '500K TF-IDF features'
    }
    return render_template('performance.html', metrics=metrics)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    print("Starting Flask application...")
    print("Make sure you have the following files in the 'model' directory:")
    print("- vectoriser-ngram-(1,2).pickle")
    print("- Sentiment-LR.pickle")
    print("- Sentiment-BNB.pickle")

    print("Loading models...")
    success = load_models()
    if not success:
        print("Warning: Models not loaded. Please ensure model files exist in the 'model' directory.")

    app.run(debug=True, host='0.0.0.0', port=5000)
