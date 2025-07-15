
#  Mithra ChatBot

**Mithra ChatBot** is a lightweight conversational AI built using a small language model architecture. It leverages Natural Language Processing (NLP) to engage in meaningful conversations based on a set of predefined intents. This project is ideal for beginner-level chatbot implementations and educational use cases.

---

##  Features

-  Small Language Model–Based Chat Engine  
-  Intent classification using deep learning (Keras/TensorFlow)  
-  Predefined intents in a structured `intents.json` format  
-  Tokenization & Lemmatization with NLTK  
-  Interactive response generation  
-  Deployable via Flask for local or web-based interfaces  
-  Publicly expose Flask app using **Ngrok**

---

##  Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras**
- **Flask**
- **NLTK**
- **NumPy / Pickle / JSON**
- **Ngrok / Pyngrok**

---

##  Project Structure

```
My_chatBot/
│
├── app.py                 # Flask API to interact with the chatbot
|__ train.py               # training of the model 
├── chatbot_model.h5       # Trained Keras model for intent classification
├── intents.json           # Intents dataset (patterns + responses)
├── words.pkl              # Preprocessed words/token list
├── classes.pkl            # Output classes from intents
├── templates/
│   └── index.html         # (Optional) Web UI template
├── static/
│   └── style.css          # CSS styling (if web UI is included)
└── README.md              # Project documentation
```

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/venkat23-hub/My_chatBot.git
cd My_chatBot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install:

```bash
pip install flask tensorflow nltk numpy pyngrok
```

### 3. Prepare NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
# save the folders in the nltk_data folder
```

### 4. Run the Chatbot

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`

---

##  Accessing via Ngrok (Optional)

If you want to make your chatbot accessible over the internet for testing or demo purposes:

### 1. Install Ngrok (if not already installed)

```bash
pip install pyngrok
```

### 2. Update your `app.py` to include:

```python
from pyngrok import ngrok
ngrok.set_auth_token("your-ngrok-auth-token")
public_url = ngrok.connect(5000)
print(" * ngrok tunnel available at:", public_url)
```

This will create a public URL to access your Flask server hosted locally.

---

##  How It Works

1. **Training:** The chatbot is trained using a basic neural network model on labeled intents.
2. **Preprocessing:** User input is tokenized and lemmatized using NLTK tools.
3. **Prediction:** The input is passed through the trained model to classify intent.
4. **Response:** A random response from the matched intent’s responses is returned.

---

##  Configuration

Ensure the following files are present before running:

- `intents.json` – Define your bot's conversation logic.
- `chatbot_model.h5` – Trained model (can be generated using a separate training script).
- `words.pkl`, `classes.pkl` – Data objects saved during preprocessing.

---

##  Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance Mithra ChatBot.

---

##  License

This project is licensed under the MIT License.

---

##  Contact

Built with ❤️ by [Venkat](https://github.com/venkat23-hub)

Feel free to reach out for feedback or collaboration!
