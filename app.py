import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import pickle
import random
import json
import os
import webbrowser
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from pyngrok import ngrok  # ✅ pyngrok handles tunneling

# === Optional: Set path to local NLTK data ===
nltk.data.path.append('./nltk_data')

# === NLP Tools ===
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

# === Load Trained Model and Data ===
model = load_model('chatbot_model.h5')

with open('intents.json', encoding='utf8') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# === NLP Preprocessing and Prediction ===
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Could you rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't know how to respond to that."

def chatbot_response(msg):
    intents_list = predict_class(msg, model)
    return get_response(intents_list, intents)

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"Health": "✅ Server is running successfully"})

@app.route("/query", methods=["GET"])
def query_chatbot():
    user_message = request.args.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response = chatbot_response(user_message)
    return jsonify({"top": {"res": response}})

# === Main Entry Point ===
if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port)  # Start ngrok tunnel
    print(f" * ngrok tunnel available at: {public_url.public_url}")

    # Automatically open browser to chatbot endpoint
    webbrowser.open(str(public_url) + "/query?message=hello")

    # Start the Flask app
    app.run(port=port)
