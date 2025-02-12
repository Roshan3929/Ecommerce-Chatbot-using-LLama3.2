import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import gensim
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import save_model
import pickle
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Constants
EMBEDDING_DIM = 100
MAX_LENGTH = 100

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_text(sen):
    """Preprocess the text data."""
    sen = re.sub('<.*?>', ' ', sen)  # Remove HTML tags
    tokens = word_tokenize(sen)  # Tokenize words
    tokens = [w.lower() for w in tokens]  # Lowercase
    table = str.maketrans('', '', string.punctuation)  # Remove punctuation
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]  # Remove non-alphabet characters
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]  # Remove stopwords
    words = [w for w in words if len(w) > 2]  # Ignore words less than 2 characters
    return words

def preprocess_data(data):
    """Preprocess the entire dataset."""
    review_lines = []
    sentences = list(data['review'])
    for sen in sentences:
        review_lines.append(preprocess_text(sen))
    return review_lines

def train_word2vec(review_lines):
    """Train a Word2Vec model on the preprocessed text."""
    model = gensim.models.Word2Vec(sentences=review_lines, vector_size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
    return model

def save_word2vec_model(model, filename):
    """Save the Word2Vec model to a file."""
    model.wv.save_word2vec_format(filename, binary=False)

def load_word2vec_embeddings(filename):
    """Load Word2Vec embeddings from a file."""
    embeddings_index = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
    return embeddings_index

def tokenize_and_pad(review_lines, max_length):
    """Tokenize and pad the sequences."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(review_lines)
    sequences = tokenizer.texts_to_sequences(review_lines)
    word_index = tokenizer.word_index
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return tokenizer, padded_sequences, word_index

def create_embedding_matrix(word_index, embeddings_index, vocab_size, embedding_dim):
    """Create an embedding matrix from the Word2Vec embeddings."""
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, index in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index, :] = embedding_vector
    return embedding_matrix

def build_lstm_model(vocab_size, embedding_dim, embedding_matrix, max_length):
    """Build an LSTM model."""
    model = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(vocab_size, embedding_dim, embedding_matrix, max_length):
    """Build a Bidirectional LSTM model."""
    model = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.6)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=5, batch_size=128, validation_split=0.2):
    """Train the model."""
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    return loss, accuracy

def predict_sentiment(model, tokenizer, text, max_length):
    """Predict sentiment for a given text."""
    filtered = [preprocess_text(text)]
    tokenize_words = tokenizer.texts_to_sequences(filtered)
    tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
    result = model.predict(tokenize_words)
    return 'positive' if result >= 0.5 else 'negative'

# Example usage
if __name__ == "__main__":
    # Load data
    data = load_data("IMDB Dataset.csv")
    
    # Preprocess data
    review_lines = preprocess_data(data)
    
    # Train Word2Vec model
    word2vec_model = train_word2vec(review_lines)
    save_word2vec_model(word2vec_model, "imdb_embedding_word2vec.txt")
    
    # Load Word2Vec embeddings
    embeddings_index = load_word2vec_embeddings("imdb_embedding_word2vec.txt")
    
    # Tokenize and pad sequences
    tokenizer, padded_sequences, word_index = tokenize_and_pad(review_lines, MAX_LENGTH)
    
    # Create embedding matrix
    vocab_size = len(word_index) + 1
    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, vocab_size, EMBEDDING_DIM)
    
    # Build and train Bidirectional LSTM model
    bilstm_model = build_bilstm_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_LENGTH)
    history2 = train_model(bilstm_model, padded_sequences, data['sentiment'].map({'positive': 1, 'negative': 0}))
    
    # Save the model
    save_model(bilstm_model, "sentiment_model.h5")

    # Save the tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Evaluate Bidirectional LSTM model
    loss, accuracy = evaluate_model(bilstm_model, padded_sequences, data['sentiment'].map({'positive': 1, 'negative': 0}))
    print(f"BiLSTM Model Accuracy: {accuracy}")
    