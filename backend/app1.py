import nltk
from nltk.chat.util import Chat
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
import re
import contractions
import joblib

# Ensure required NLTK resources are downloaded
nltk.download('stopwords')

# Define the reflections dictionary
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

# Define the dialogue pairs in English
pairs = [
    (r"hello", ["Hello! How are you feeling today?"]),
    (r"hi", ["Hi! How are you feeling today?"]),
    (r"hey", ["Hey! How's it going?"]),
    (r"how are you", ["I'm glad to hear that! What did you do today?"]),
    (r"what can you do", ["I can chat with you and help you feel better."]),
    (r"who are you", ["I am your Calminds assistant designed to listen and help you."]),
    (r"what is your name", ["I am your Calminds assistant."]),
    (r"thank you", ["You're welcome! If you need to talk, I'm here."]),
    (r"goodbye", ["Goodbye! Have a great day!"]),
    (r"I need help", ["I'm here to listen. Can you give me more details about what's wrong?"]),
    (r".sad.", ["I'm sorry you're feeling sad. Do you want to talk about it?"]),
    (r".bad.", ["I'm sorry you're feeling bad. Do you want to talk about it?"]),
    (r".happy.", ["That's great to hear! Can you share what made your day special?"]),
    (r".joyful.", ["That's great to hear! Can you share what made your day special?"]),
    (r"tell me about yourself", ["I am a chatbot designed to help people talk about their feelings."]),
    (r"how old are you", ["I don't have an age, but I'm here to help!"]),
    (r".chatbot.", ["Yes, I am a chatbot. How can I help you today?"]),
    (r".solutions.", ['Of course! You can view the solutions by <a href="/solutions.html">clicking this link</a>']),
    (r".tired.", ["I'm sorry you're feeling negative. Remember, it's important to seek professional help."]),
    (r".sick.", ["I'm sorry you're feeling negative. Remember, it's important to seek professional help."]),
]

def load_model(model_filename="model/sentiment_modelSVM.pkl"):
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except FileNotFoundError:
        print(f"The model file {model_filename} is not found.")
        return None

def formatText(text):
    try:
        text = contractions.fix(text)
        text = re.sub(r'[^\w\s]|[\d]', '', text.lower().strip())
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        return text
    except Exception as e:
        print(f"Error formatting text: {e}")
        return ""

def predict_sentiment(text, model):
    try:
        text_processed = formatText(text)
        text_tfidf = model['tfidf'].transform([text_processed])
        prediction = model['clf'].predict(text_tfidf)
        if prediction[0] == 'positive':
            return 0
        else:
            return 1
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return None

def suggest_solution(emotion):
    if emotion == 0:
        return "It's great that you're feeling good! Keep doing what makes you happy."
    elif emotion == 1:
        return "I'm sorry you're feeling bad. You could try talking to a friend, taking a walk, or consulting a professional."
    else:
        return "I couldn't determine your sentiment. Can you rephrase?"

def get_response(emotion, user_input):
    if emotion == 0:
        return "It's great that you're feeling positive! Can you share what made your day special?"
    elif emotion == 1:
        return "I'm sorry you're feeling negative. Remember, it's important to seek professional help."
    else:
        return "I couldn't determine your sentiment. Can you rephrase?"

# Create the chatbot
chat = Chat(pairs, reflections)
model = load_model()

# Create a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return render_template("chatbot.html")

@app.route('/chat', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({"response": "No message provided."}), 400

        if model is None:
            return jsonify({"response": "Sentiment model not found."}), 500

        response = chat.respond(user_input)
        if response:
            return jsonify({"response": response, "solution": ""})

        predicted_emotion = predict_sentiment(user_input, model)
        if predicted_emotion is None:
            return jsonify({"response": "Error predicting sentiment."}), 500

        response = get_response(predicted_emotion, user_input)
        solution = suggest_solution(predicted_emotion)
        return jsonify({"response": response, "solution": solution})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": f"Error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
