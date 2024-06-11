

# Toxic Comment Detection API

This project implements a machine learning model to detect toxic comments using TensorFlow and serves predictions through a FastAPI web service. The model is trained to identify various types of toxicity, including toxic, severe toxic, obscene, threat, insult, and identity hate.

### Prerequisites

- Python 3.9 (preferably, inorder to match the version of tensorflow used)
- Pip (Python package installer)

### Installing

1. **Clone the repository:**

```sh
git clone https://github.com/your-username/toxic-comment-detection.git
cd toxic-comment-detection
```

2. **Create a virtual environment and activate it:**

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install the dependencies:**

```sh
pip install -r requirements.txt
```

### Running the FastAPI Application

1. **Ensure `toxicity_model.h5` and `tokenizer.pickle` are in the project directory.**
   
2. **Start the FastAPI server:**

```sh
uvicorn main:app --reload
```

3. **Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the API documentation and test the endpoints.**

### API Usage

#### Predict Toxicity

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**
  
  ```json
  {
    "text": "Your comment text here."
  }
  ```

- **Response:**
  
  ```json
  {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  }
  ```

### Training the Model

The model and tokenizer were trained in a Google Colab environment. Here’s how you can train and save the model and tokenizer:

1. **Train the Model and Tokenizer:**

   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Embedding, LSTM, Dense
   import pickle

   # Sample training data
   training_data = ["This is a sample comment.", "Another comment."]
   labels = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]  # Sample labels

   # Initialize and fit the tokenizer
   tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
   tokenizer.fit_on_texts(training_data)

   # Save the tokenizer
   with open('tokenizer.pickle', 'wb') as handle:
       pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

   # Convert text to sequences
   sequences = tokenizer.texts_to_sequences(training_data)
   # Pad sequences
   padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

   # Define the model
   model = Sequential([
       Embedding(20000, 128, input_length=100),
       LSTM(64, return_sequences=True),
       LSTM(64),
       Dense(6, activation='sigmoid')
   ])

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(padded_sequences, np.array(labels), epochs=10)

   # Save the model
   model.save('toxicity_model.h5')
   ```

2. **Download the model and tokenizer:**

   ```python
   from google.colab import files

   files.download('toxicity_model.h5')
   files.download('tokenizer.pickle')
   ```

