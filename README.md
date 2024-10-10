This repository contains the implementation of a Sentiment Analysis Model using LSTM (Long Short-Term Memory) neural networks in TensorFlow:

Purpose: Classifies text into sentiment categories: positive, negative, and neutral.

Dataset:
Format: CSV
Columns:
text: Textual data to analyze
sentiment: Labeled sentiments

Model Architecture:
Embedding Layer: Converts input tokens to dense vectors.
LSTM Layers: Two stacked LSTM layers with 64 and 32 units, respectively.
Dense Layers: Fully connected layers with ReLU activation followed by a softmax layer for multi-class classification.

Preprocessing:
Tokenization and padding to a fixed sequence length of 100 tokens.
Max vocabulary size: 5000 words.
Sentiment labels converted to numeric values:
negative: 0
positive: 1
neutral: 2

Training:
Optimizer: Adam
Loss Function: Sparse categorical cross-entropy
Data Split: 80% for training and 20% for validation.

Dependencies:
Python 3.x
TensorFlow 2.x
Pandas
NumPy
Scikit-learn

Running the Model:
Place the dataset file (newtest.csv) in the project directory.
Execute the script using Python.

Evaluation: Validation accuracy is displayed after training.
Future Improvements:
Fine-tuning model architecture for better accuracy.
Increasing dataset size.
Experimenting with different word embedding methods (e.g., GloVe, Word2Vec).

Contributions: Contributions are welcome.
