import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import pickle
import random
import json
import os
import webbrowser
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from pyngrok import ngrok

nltk.data.path.append('./nltk_data')
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
model = load_model('chatbot_model.h5')

with open('intents.json', encoding='utf8') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

chat_history = []

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
# Predict the class of the sentence
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

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    response = ""
    if request.method == "POST":
        user_message = request.form.get("message")
        if user_message:
            response = chatbot_response(user_message)
            chat_history.append((user_message, response))
    return render_template("index.html", response=response, chat_history=chat_history)

@app.route("/query", methods=["GET"])
def query_chatbot():
    user_message = request.args.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response = chatbot_response(user_message)
    chat_history.append((user_message, response))
    return jsonify({"top": {"res": response}})

@app.route("/save_chat", methods=["GET"])
def save_chat():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Chat Transcript", ln=True, align='C')
    pdf.ln(10)

    for user_msg, bot_msg in chat_history:
        pdf.multi_cell(0, 10, f"You: {user_msg}")
        pdf.multi_cell(0, 10, f"Bot: {bot_msg}")
        pdf.ln(5)

    os.makedirs("downloads", exist_ok=True)
    filepath = os.path.join("downloads", "chat_history.pdf")
    pdf.output(filepath)
    return jsonify({"status": "saved", "path": filepath})

@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel available at: {public_url.public_url}")
    webbrowser.open(str(public_url.public_url))
    app.run(port=port)