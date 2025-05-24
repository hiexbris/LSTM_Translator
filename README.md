# LSTM_Translator

A neural machine translation system that translates French sentences into English using a Seq2Seq architecture with Bahdanau Attention. This project demonstrates a complete pipeline from preprocessing and embedding to training, evaluation, and visualization through a Pygame interface.

---

## 🚀 Overview

This project implements a **French-to-English machine translation** model using a **Bidirectional LSTM Encoder**, **Bahdanau Attention**, and an **LSTM Decoder**. It leverages **pretrained FastText word embeddings** (`cc.fr.300.vec.gz` and `cc.en.300.vec.gz`) to provide meaningful semantic initialization and better generalization.

---

## 📁 Project Structure

```
LSTM_Machine_Translation/
│
├── dataset_class.py # Builds vocab and embedding matrix from FastText
├── model.py # Encoder-Decoder with Bahdanau Attention
├── testing_single_data.py # Prepares training data using vocab and embeddings
├── testing_data.py # Prepares testing data in same format
├── app.py # Pygame app to visualize / test translation, includes attention heatmap display
├── README.md # This file
└── .gitignore # Ignores large .vec.gz files
```


## 🧠 Model Architecture

### 🔹 Encoder
- A **Bidirectional LSTM** is used to capture both forward and backward context of the French input sentence.
- The hidden states from both directions are concatenated and passed to the decoder.

### 🔹 Bahdanau Attention
- Instead of just relying on the final encoder state, Bahdanau Attention (also known as additive attention) computes a **context vector** at each decoding step by attending to **all** encoder hidden states.
- This allows the decoder to dynamically focus on relevant parts of the input sentence during translation.
- It improves translation quality especially for longer or more complex input sentences.

### 🔹 Decoder
- The decoder is an **LSTM** that generates one word at a time.
- It takes in the previous word, previous hidden state, and the attention-generated context vector to produce the next word.
- The decoder is trained using **teacher forcing** for stability.

---

## 📦 FastText Embeddings

We use:
- `cc.fr.300.vec.gz` for French word vectors
- `cc.en.300.vec.gz` for English word vectors
- Due to big sizes couldn't be added here.

---

## 🧾 Data Preparation

- `dataset_class.py`:  
  Reads text data, tokenizes, builds vocab, and loads FastText vectors to create the **embedding matrix**.

- `testing_single_dataset.py`:  
  Uses the vocab and embedding matrix to tokenize, numericalize, pad, and batch training data.

- `testing_data.py`:  
  Prepares testing samples similarly for inference/evaluation.

---

## 🎮 Pygame Interface (`app.py`)

A simple Pygame-based GUI to type a French sentence and see the English translation predicted by the model. It also displays a visual attention heatmap, showing which parts of the input the model focuses on at each step—making the project both interactive and interpretable.

---

## 🛠 Training (Included in main.py)

- Use `dataset_class.py` for loading training batches, embedding matrix, and vocabulary
- Pass through `model.py`’s encoder-decoder
- Use cross-entropy loss and Adam optimizer

---

Notes

This uses Python 3.11 as the newer versions aren't compatible with CUDA Torch needed for GPU usage while training
Checkpoints file saves the models every 5 epoches and the best model yet
For training, The model requires two embeddings French (cc.fr.300.vec.gz) and English (cc.en.300.vec.gz). These are fasttext embeddings any other pretrained can be used as well
