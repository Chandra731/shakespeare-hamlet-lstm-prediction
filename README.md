
# Next Word Prediction using LSTM and Streamlit

This project demonstrates a next-word prediction system using an LSTM neural network trained on Shakespeare's *Hamlet*. The system predicts the next word based on an input sentence and is deployed as a web application using Streamlit.

## Features
- Trained on Shakespeare's *Hamlet* text.
- Allows users to input a sentence and predicts the next word.
- Built using LSTM with TensorFlow and Keras.
- User-friendly web interface created with Streamlit.

## Installation

1. Clone this repository:

   `git clone https://github.com/Chandra731/shakespeare-hamlet-lstm-prediction
   cd next-word-prediction`

2. Install the required dependencies:

   `pip install -r requirements.txt`

3. Run the Streamlit app:

   `streamlit run app.py`

## Model Training
The LSTM model was trained on the *Hamlet* text from the NLTK Gutenberg corpus. The dataset was preprocessed by tokenizing the text into sequences and padded for input into the LSTM model. After training, the model and tokenizer were saved for future use.

## App Features

- **Input**: The user enters a sentence (e.g., "Now is the winter of our").
- **Prediction**: The app uses the trained LSTM model to predict the next word in the sequence.
- **Display**: The predicted word is displayed on the app.

## Example Usage

1. Input: `To be or not to be`
2. Output: `that` (the predicted next word)

## License

This project is licensed under the MIT License.
