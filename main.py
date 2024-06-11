from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model('toxicity_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = FastAPI()

class Comment(BaseModel):
    text: str

@app.post('/predict')
def predict(comment: Comment):
    sequence = tokenizer.texts_to_sequences([comment.text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    
    prediction = model.predict(padded_sequence)
    
    result = {
        'toxic': 1 if prediction[0][0] > 0.5 else 0,
        'severe_toxic': 1 if prediction[0][1] > 0.5 else 0,
        'obscene': 1 if prediction[0][2] > 0.5 else 0,
        'threat': 1 if prediction[0][3] > 0.5 else 0,
        'insult': 1 if prediction[0][4] > 0.5 else 0,
        'identity_hate': 1 if prediction[0][5] > 0.5 else 0
    }
    
    return result
