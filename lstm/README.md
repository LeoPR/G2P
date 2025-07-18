# PyTorch G2P: LSTM-based Grapheme-to-Phoneme Model

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository provides a clean, well-documented, and educational implementation of a Grapheme-to-Phoneme (G2P) conversion system using an LSTM-based sequence-to-sequence architecture in PyTorch.

The goal is to offer a robust, easy-to-understand codebase for researchers, students, and practitioners interested in sequence modeling and speech synthesis, with minimal external dependencies.

## Features

*   **Classic Seq2Seq Architecture**: A robust implementation of an LSTM-based Encoder-Decoder model, a proven architecture for sequence-to-sequence tasks.
*   **Multi-Language Support**: Easily handles multiple languages in a single model by using language-specific tags (e.g., `<en-us>:`, `<pt-br>:`) in the input.
*   **Phonetically-Aware Embeddings**: An optional feature to initialize the decoder's embedding layer with phonetic features, injecting linguistic knowledge to potentially speed up convergence and improve accuracy.
*   **Intelligent Data Splitting**: Includes a stratified data splitting strategy to ensure that rare phonemes and n-grams are appropriately represented in both training and validation sets, leading to more reliable evaluation.
*   **Clean and Modular Code**: Clear separation of concerns:
    *   `G2P.py`: Model architecture (Encoder, Decoder, Seq2Seq).
    *   `G2P_utils.py`: Data handling, vocabulary, and helper functions.
    *   `G2P_train.py`: The training and model-saving script.
    *   `G2P_inference.py`: A script for evaluation and simple API-like usage.
*   **Performance Evaluation**: Built-in script to calculate Word Accuracy (WA) and Phoneme Error Rate (PER) and display qualitative examples of model predictions.

## Architecture: LSTM-based Seq2Seq

Grapheme-to-Phoneme conversion is a classic sequence-to-sequence task: it maps an input sequence (characters/graphemes) to an output sequence (phonemes). This project uses a Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN) perfectly suited for this.

*   **How it Works**: An **Encoder LSTM** reads the entire grapheme sequence (e.g., `c`, `a`, `t`) token-by-token and compresses its meaning into a fixed-size "context vector" (the final hidden and cell states). A **Decoder LSTM** then takes this context vector and generates the phoneme sequence one phoneme at a time, using its own previously generated phoneme as input for the next step.

*   **Why use an LSTM for G2P?**
    *   **Efficiency**: LSTMs are generally less parameter-heavy than more modern architectures and can train effectively on smaller datasets and with less computational hardware.
    *   **Sequential Intuition**: The step-by-step nature of RNNs is a very natural fit for sequence generation tasks like language and phoneme prediction.
    *   **Strong Baseline**: They are a proven technology and serve as a robust baseline to measure the benefits of more complex architectures.

*   **Limitations**:
    *   **Information Bottleneck**: All information from the input sequence must be compressed into one fixed-size context vector, which can be a limitation for very long or complex words.
    *   **Sequential Computation**: The model cannot be fully parallelized during training, as each step `t` depends on the completion of step `t-1`.

### A Note on the Transformer Model

In the parent repository, a **Transformer-based model** is also provided for comparison. Unlike the LSTM, the Transformer eschews recurrence and relies entirely on **self-attention mechanisms**. This allows it to process all input tokens in parallel, making training significantly faster on GPUs and often leading to superior performance on large datasets by better modeling long-range dependencies. This LSTM implementation serves as a powerful and efficient alternative, especially when training data or computational resources are limited.

## Getting Started

Follow these steps to prepare your data, train a new model, and run inference.

### 1. Prerequisites

It is recommended to use a Python virtual environment. Install the required packages using the provided file.

```bash
# It's good practice to upgrade pip
python -m pip install -U pip

# Install all dependencies
pip install -r requirements.txt
```

## 2. Data Preparation
Your training data must be in a tab-separated (.tsv) file with one grapheme<TAB>phoneme pair per line.
```
Example pt-br.tsv:
Generated tsv
casa	'ka.zɐ
gato	'ga.tu
```

Tsv
The training script automatically detects all .tsv files in a ./dicts directory and uses the filename to create a language tag. For example, pt-br.tsv will cause all words from that file to be prefixed with <pt-br>: during training.

## 3. Training the Model
To train the model using all .tsv files found in the ./dicts directory, simply run the training script.
Generated bash
python G2P_train.py
Use code with caution.
Bash
The script will use the "intelligent" stratified split, train for the specified number of epochs, and save the best-performing model checkpoint as g2p_model.pth. A training_log.txt file will also be created.

## 4. Inference and Evaluation
After training, you can use the saved model to convert new words or evaluate its performance.
Full Evaluation
The G2P_inference.py script calculates performance metrics (Word Accuracy, Phoneme Error Rate) on a test file and shows qualitative examples.
Generated bash
python G2P_inference.py --model_path g2p_model.pth --data_path ./dicts/pt-br.tsv --lang pt-br
Use code with caution.
Bash
Simple API-like Usage
For quick, single-word conversions without the full evaluation overhead, use the G2P_inference_minimal.py script. This is ideal for integration into other applications.
Generated bash
# The script has a built-in example usage
python G2P_inference_minimal.py
Use code with caution.
Bash
License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.