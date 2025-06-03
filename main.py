import json


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


TESTING_SIZE =  28619 / 2


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







#for item in datastore:
    #sentences.append(item['headline'])
    #labels.append(item['is_sarcastic'])
    #urls.append(item['article_link'])

tokenizer = Tokenizer(num_words = 50, oov_token = '<OOOv>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


sequences = tokenizer.texts_to_sequences(sentences)



padded_sequences = pad_sequences(sequences, padding='post')
print(padded_sequences[0])
print(padded_sequences.shape)

training_size = 28619 / 2


training_sentences = sentences[training_size:]
training_labels = labels[training_size:]

testing_sentences = sentences[:training_size]
testing_labels = labels[:training_size]

