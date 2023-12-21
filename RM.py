

import random
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import RegexpTokenizer
import random

df = pd.read_csv("Book1.csv")

input_data = list(df['Paragraph'].values)


text = "".join(input_data)


text_length = text[:10000]

# Tokenization
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(text_length.lower())

# Unique tokens and their index
unique_tokens = np.unique(tokens)
unique_tokens_index = {token: idx for idx, token in enumerate(unique_tokens)}

n_words = 10
input_word = []
next_word = []

# Create input and output sequences
for i in range(len(tokens) - n_words):
    input_word.append(tokens[i:i + n_words])
    next_word.append(tokens[i + n_words])

x = np.zeros((len(input_word), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)

for i, words in enumerate(input_word):
    for j, word in enumerate(words):
        x[i, j, unique_tokens_index[word]] = 1
    y[i, unique_tokens_index[next_word[i]]] = 1


model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens)))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])


model.fit(x, y, batch_size=128, epochs=30, shuffle=True)

# Define the predict_next_word function
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        x[0, i, unique_tokens_index[word]] = 1

    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]


def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    word_tokens = []
    for _ in range(text_length):
        subsequence = "".join(tokenizer.tokenize("".join(word_sequence).lower())[current: current + n_words])

        try:
            choice = unique_tokens[random.choice(predict_next_word(subsequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_tokens.append(choice)
        current += 1

    return " ".join(word_tokens)

# Generate text
generated_text = generate_text("A precious gift for my siblings", 100, 5)
print(generated_text)

model = load_model("pre_trained_models/model.h5")