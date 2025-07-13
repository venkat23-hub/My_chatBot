import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import pickle
import random
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

# === Download NLTK resources (for Render and local use) ===
nltk.download('punkt')
nltk.download('wordnet')

# === Initialize Preprocessing Tools ===
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

# === Load Model and Required Files ===
model = load_model('chatbot_model.h5')

try:
    with open('intents.json', encoding='utf8') as f:
        intents = json.load(f)
except Exception as e:
    raise RuntimeError("❌ Failed to load intents.json: " + str(e))

try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except Exception as e:
    raise RuntimeError("❌ Failed to load pickle files: " + str(e))

# === NLP Helpers ===
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
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

def decrypt(msg):
    return msg.replace("+", " ")

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)  # ✅ Allow requests from frontend (React or browser)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"Health": "✅ Server is running successfully"})

@app.route("/query/<message>", methods=["GET"])
def query_chatbot(message):
    user_message = decrypt(message)
    response = chatbot_response(user_message)
    return jsonify({"top": {"res": response}})

# === Entry Point for Development ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
