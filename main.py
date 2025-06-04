import json


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


vocab_size = 10000
embedding_dimension = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'

TESTING_SIZE =  28619 // 2


sentences = []
labels = []
urls = []


num_lines = 0

with open("Sarcasm.json", 'r') as file:

    for line in file:
        if num_lines >= TESTING_SIZE:
            break
        curr_line = json.loads(line)
        sentences.append(curr_line["headline"])
        labels.append(curr_line["is_sarcastic"])
        urls.append(curr_line["article_link"])
        num_lines += 1






#the first half of the array should be training data
training_sentences = sentences[:TESTING_SIZE]
training_labels = labels[:TESTING_SIZE]


#the second half of the array should be testing on the trained data
testing_sentences = sentences[TESTING_SIZE:]
testing_labels = labels[TESTING_SIZE:]






tokenizer = Tokenizer(num_words = vocab_size, oov_token = '<OOOv>')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index


training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)




model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dimension),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()


num_epochs = 30
history = model.fit(
    training_padded,
    training_labels,epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2 )
