import random
import json
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# Initialize
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

words = []
classes = []
documents = []
ignore_words = ['!', '?', '$', '@']

# Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Tokenize and preprocess patterns
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = tokenizer.tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Print summary
print(len(documents), "Documents")
print(len(classes), "Classes:", classes)
print(len(words), "Unique lemmatized words")

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to numpy
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train
model.fit(train_x, train_y, epochs=200, batch_size=10, verbose=1)

# Save model and data
model.save('chatbot_model.h5')
print("Model saved to chatbot_model.h5")

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Words and classes saved (words.pkl, classes.pkl)")
