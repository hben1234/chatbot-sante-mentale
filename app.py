from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.chat.util import Chat
from nltk.corpus import stopwords
import re
import contractions
import joblib

nltk.download('stopwords')

# Create a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Define the reflections dictionary
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
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
pairs =[
    (r"hello|hi|hey", ["Hello! How are you feeling today?"]),
    (r"how are you(.*)", ["I'm here to listen. How can I support you today?"]),
    (r"(.*)chatbot(.*)", ["Yes, I'm here to help. What's on your mind?"]),
    (r"what can you do(.*)|who are you(.*)|tell me about yourself(.*)|what is your name", ["I'm your supportive assistant here to chat and help you."]),
    (r"thank you|thanks", ["You're welcome! I'm here for you whenever you need."]),
    (r"goodbye|bye|see you", ["Goodbye! Take care of yourself."]),
    (r"I need help(.*)|help me(.*)|can you help me(.*)", ["Of course. What's on your mind?"]),
    (r"i am (.*)|i'm (.*)", ["Why do you feel %1?", "How long have you been feeling %1?"]),
    (r"I'm feeling overwhelmed", ["It sounds like things are tough right now. Would you like to talk about it?"]),
    (r"I'm feeling stuck in my life", ["What's holding you back right now?"]),
    (r"I'm feeling lost", ["Feeling lost is hard. What's something that usually helps you find direction?"]),
    (r"I'm feeling anxious", ["Anxiety can be overwhelming. What's been causing your anxiety?"]),
    (r"I'm feeling scared", ["It's okay to feel scared. What's making you feel this way?"]),
    (r"I'm feeling confused", ["Confusion happens. What's causing confusion for you right now?"]),
    (r"I'm feeling discouraged", ["It's normal to feel this way sometimes. What usually lifts your spirits?"]),
    (r"I'm feeling frustrated", ["I hear you. What's causing your frustration?"]),
    (r"I'm feeling jealous", ["Jealousy can be tough. What's been triggering these feelings?"]),
    (r"I'm feeling lonely", ["Loneliness is hard. Is there something specific you'd like to talk about?"]),
    (r"I'm feeling hurt", ["I'm sorry to hear that. What's been hurting you?"]),
    (r"I'm feeling guilty", ["Guilt is tough. What's weighing on your mind?"]),
    (r"I'm feeling ashamed", ["I'm here to listen. What's been on your mind?"]),
    (r"I'm feeling embarrassed", ["It's okay to feel embarrassed. What's been on your mind?"]),
    (r"I'm feeling grateful", ["Gratitude is a wonderful feeling. What's been making you feel grateful?"]),
    (r"I'm feeling inspired", ["That's great to hear! What's been inspiring you lately?"]),
    (r"I'm feeling motivated", ["Motivation is key. What's been keeping you motivated?"]),
    (r"I'm feeling determined", ["That's great to hear! What's motivating your determination?"]),
    (r"I'm feeling hopeful", ["Hope is powerful. What's giving you hope right now?"]),
    (r"I'm feeling optimistic", ["Optimism is refreshing. What's been making you feel optimistic?"]),
    (r"I'm feeling supportive", ["That's wonderful! How can I support you today?"]),
    (r"I'm feeling loving", ["Love is beautiful. How can I assist you today?"]),
    (r"I'm feeling empathetic", ["Empathy is important. How can I be there for you today?"]),
    (r"I'm feeling kind", ["Kindness matters. How can I support you today?"]),
    (r"I'm feeling generous", ["Generosity is admirable. How can I assist you today?"]),
    (r"I'm feeling compassionate", ["Compassion is key. How can I be there for you today?"]),
    (r"I'm feeling forgiving", ["Forgiveness is liberating. How can I support you today?"]),
    (r"I'm feeling understanding", ["Understanding is valuable. How can I be there for you today?"]),
    (r"I'm feeling wise", ["Wisdom is powerful. How can I assist you today?"]),
    (r"I'm feeling enlightened", ["That's wonderful! How can I assist you today?"]),
    (r"I'm feeling powerful", ["Power comes in many forms. How can I support you today?"]),
    (r"I'm feeling strong", ["Strength is admirable. How can I be there for you today?"]),
    (r"I'm feeling resilient", ["Resilience is important. How can I support you today?"]),
    (r"I'm feeling confident", ["Confidence is empowering. How can I be there for you today?"]),
    (r"I'm feeling brave", ["Bravery is commendable. How can I support you today?"]),
    (r"how can I feel better|what should I do", ["You could try taking a deep breath, talking to a friend, or doing something you enjoy."]),
    (r"give me advice|what can I do", ["Remember to take care of yourself. It's okay to seek support from others."]),
    (r"I'm feeling courageous", ["That's wonderful to hear! What's been boosting your courage?"]),
    (r"I'm feeling focused", ["That's great! What's helping you stay focused?"]),
    (r"I'm feeling alert", ["Being alert is important. What's been keeping you alert?"]),
    (r"I'm feeling attentive", ["That's good to hear! What's been keeping you attentive?"]),
    (r"I'm feeling aware", ["Awareness is valuable. What's been keeping you aware?"]),
    (r"I'm feeling attuned", ["That's great! What's been helping you stay attuned?"]),
    (r"I'm feeling responsive", ["That's good to hear! What's been making you responsive?"]),
    (r"i feel (.*)", ["What's been causing you to feel %1?", "How can I support you with feeling %1?"]),
    (r"(.*)sad(.*)|(.*)bad(.*)|(.*)unhappy(.*)", ["I'm sorry to hear that. It's okay to feel this way. Would you like to talk about it?"]),
    (r"(.*)happy|(.*)good(.*)|(.*)joyful(.*)", ["That's wonderful to hear! "]),
    (r".*happy", [" That's wonderful to hear!"]),
    (r"sad|bad|unhappy", ["I'm sorry to hear that. It's okay to feel this way. Would you like to talk about it?"]),
    (r"happy|good|joyful", ["That's wonderful to hear! What's making you feel %1?"]),
    (r"yes", ["that's great ,if you want an emergency intervention you should book an appointment on the appointment page and check each session price for more details"]),
    (r"no", ["I'm sorry to hear that. Is there anything else I can help you with?"]),
    (r".*advice", ["that's great ,You are not alone. So many people experience mental health challenges, and seeking help is a sign of strength "]),
    (r"(.*)calm(.*)", ["I'm glad you're feeling calm. How can I help you maintain this calm?"]),
    (r"(.*)angry(.*)", ["I'm here to listen. What's been making you feel angry?"]),
    (r"(.*)sick(.*)", ["I'm sorry to hear that you're feeling sick. Is there anything specific you'd like to talk about?"]),
    (r"(.*)tired(.*)", ["I'm sorry to hear that you're feeling tired. Is there anything specific you'd like to talk about?"]),
    (r"thank you|thanks", ["You're welcome! I'm here for you whenever you need."]),
    (r"(.*)support", ["I'm here to support you whenever you need. How can I assist you today?"]),
    (r".*help", ["I'm here to help you. Try going to bed and waking up at consistent times, even on weekends, to regulate your body's natural sleep-wake cycle "]),
    (r".*sleep", [" Getting enough sleep is so important for our mental health. How can I help you establish a better sleep routine? "]),
    (r".*sleeping", ["just breathe deeply and focus on your breathing while counting until you fall asleep"]),
    (r".*solutions.*", ["Of course!  Have you ever considered meditation? There are some great free guided meditations online"]),
    (r"can you be my friend?", ["Of course! I'm here for you."]),
    (r"i don't have friends", ["You're not alone. I'm here to chat anytime you need."]),
    (r"i need help", ["I'm here to help you. Try going to bed and waking up at consistent times, even on weekends, to regulate your body's natural sleep-wake cycle "]),
    
    (r"(.*)joke", ["Sure! Here's a joke: Two fish are in a tank. One turns to the other and says, 'Do you know how to drive this thing?'"]),
    (r"(.*)", ["Could you tell me more about that?", "I'm here to listen. Please share more details."]),
]


# Function to load the model
def load_model(model_filename="backend/model/sentiment_modelSVM.pkl"):
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except FileNotFoundError:
        print(f"The model file {model_filename} is not found.")
        return None

# Function to format text
def format_text(text):
    try:
        text = contractions.fix(text)
        print(f"Text after contractions fix: {text}")
        text = re.sub(r'[^\w\s]|[\d]', '', text.lower().strip())
        print(f"Text after removing punctuation and digits: {text}")
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
        print(f"Text after removing stopwords: {text}")
        return text
    except Exception as e:
        print(f"Error formatting text: {e}")
        return ""

# Function to predict sentiment
def predict_sentiment(text, model):
    try:
        text_processed = format_text(text)
        text_tfidf = model['tfidf'].transform([text_processed])
        prediction = model['clf'].predict(text_tfidf)
        print(f"Prediction: {prediction}")
        return 0 if prediction[0] == 'positive' else 1
    except Exception as e:
        print(f"Error predicting sentiment: {e}")
        return None

# Function to suggest a solution
def suggest_solution(emotion):
    if emotion == 0:
        return "It's great that you're feeling good! Keep doing what makes you happy."
    elif emotion == 1:
        return "I'm sorry you're feeling bad. You could try talking to a friend, taking a walk, or consulting a professional."
    else:
        return "I couldn't determine your sentiment. Can you rephrase?"


# Function to get a response based on emotion
def get_response(emotion, user_input):
    if emotion == 0:
        return "It's great that you're feeling positive! Can you share what made your day special?"
    elif emotion == 1:
        return "I'm sorry you're feeling bad. You could try talking to a friend, taking a walk, or consulting a professional. What's troubling you?"
    else:
        return "I couldn't determine your sentiment. Can you rephrase?"



# Create the chatbot
chat = Chat(pairs, reflections)
model = load_model()

@app.route('/')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/chatbot.html')
def chatbot_page():
    return render_template('chatbot.html')
@app.route('/about.html')
def about():
    return render_template('about.html')
@app.route('/appointment.html')
def appointment():
    return render_template('appointment.html')
@app.route('/blog.html')
def blog():
    return render_template('blog.html')
@app.route('/contact.html')
def contact():
    return render_template('contact.html')
@app.route('/detail.html')
def detail():
    return render_template('detail.html')
@app.route('/testimonial.html')
def testimonial():
    return render_template('testimonial.html')
@app.route('/price.html')
def price():
    return render_template('price.html')
@app.route('/register.html')
def register():
    return render_template('register.html')
@app.route('/search.html')
def search():
    return render_template('search.html')
@app.route('/service.html')
def service():
    return render_template('service.html')
@app.route('/team.html')
def team():
    return render_template('team.html')
@app.route('/help.html')
def help():
    return render_template('help.html')

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

        # Modify response to include HTML link
        if "solutions" in user_input:
            response += ' You can view the solutions by <a href="/help.html">clicking this link</a>.'

        return jsonify({"response": response, "solution": solution})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": f"Error: {e}"}), 500




if __name__ == '__main__':
    app.run(debug=True, port=8000)