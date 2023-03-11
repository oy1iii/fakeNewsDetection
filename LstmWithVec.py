import numpy as np
import pandas as pd
import nltk
import re
import gensim
import pickle
import DataPreProcess
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gensim.models.keyedvectors import KeyedVectors

y = DataPreProcess.train_df["label"]
x = DataPreProcess.train_df["text"]

y = y.apply(lambda x: 0 if x == 1 else 1)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(x)

x = tokenizer.texts_to_sequences(x)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

maxlength = 700

x = pad_sequences(x, maxlen=maxlength)

vector_size = 300
min_count = 1
# word2Vec_model = gensim.models.Word2Vec(sentences=X, size=vector_size, window=5, min_count=min_count)

g_w2v_model_file = 'GoogleNews-vectors-negative300.bin'
word2Vec_model = KeyedVectors.load_word2vec_format(g_w2v_model_file, binary=True)

# add index for null text
vocab_size = len(tokenizer.word_index) + 1

weight_matrix = np.zeros((vocab_size, vector_size))
for word, i in tokenizer.word_index.items():
    try:
        weight_matrix[i] = word2Vec_model[word]
    except KeyError:
        continue
    # weight_matrix[i] = word2Vec_model[word]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=24, shuffle =True)

model = Sequential()
model.add(Embedding(vocab_size, output_dim=vector_size, weights=[weight_matrix], input_length=maxlength, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, validation_split=0.3, epochs=12)

model.save('model_Vec_Lsmt.h5')

y_pred = (model.predict(x_test) > 0.5).astype("int")

print(accuracy_score(y_test, y_pred))