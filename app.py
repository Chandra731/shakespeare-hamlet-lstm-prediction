import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
model = load_model('my_model.keras')

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the function to predict the next word
def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title('Next Word Prediction App')
st.write('Enter a sentence below to predict the next word.')

# User input for the sentence
input_text = st.text_input('Enter a sentence:', 'Now is the winter of our')

# Maximum sequence length
max_sequence_len = model.input_shape[1] + 1

# Predict the next word when the user submits a sentence
if st.button('Predict'):
    next_word = predict_next_word(input_text, model, tokenizer, max_sequence_len)
    if next_word:
        st.write(f'The next word might be: **{next_word}**')
    else:
        st.write('Could not predict the next word. Try another sentence.')
