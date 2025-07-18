# PyTorch G2P: A Comparative Implementation of LSTM and Transformer Models
[English](./README.md) | [Português](./README.pt.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository provides a clear and educational implementation of a Grapheme-to-Phoneme (G2P) conversion system in PyTorch. It is designed as a practical, head-to-head comparison between two cornerstone sequence-to-sequence architectures:

1.  A traditional **LSTM-based model** (`/lstm`)
2.  A modern **Transformer-based model** (`/transformer`)

The goal is to offer a clean, well-documented, and minimal-dependency codebase for researchers, students, and practitioners interested in sequence modeling and speech synthesis.

## Features

*   **Dual Architectures**: Self-contained implementations of both LSTM and Transformer models for direct performance and architectural comparison.
*   **Phonetically-Aware Embeddings**: An optional feature to initialize the decoder's embedding layer with phonetic features, potentially speeding up convergence and improving accuracy.
*   **Robust Data Handling**: Includes an intelligent stratified data splitting strategy to ensure that rare phonemes and n-grams are appropriately represented in both training and validation sets.
*   **Clean and Modular Code**: Clear separation of model definition (`G2P.py`), training (`G2P_train.py`), and inference (`G2P_inference.py`) for ease of understanding and modification.
*   **Performance Evaluation**: Built-in script to calculate Word Accuracy (WA) and Phoneme Error Rate (PER) and display qualitative examples.

<!-- You can add a high-level diagram of the G2P process here -->
<!-- Example: <p align="center"><img src="path/to/your/diagram.png" width="700"></p> -->

## Architectures: LSTM vs. Transformer for G2P

Grapheme-to-Phoneme conversion is a classic sequence-to-sequence task: it maps an input sequence (characters) to an output sequence (phonemes). The choice of architecture has significant implications for performance, training speed, and data requirements.

### LSTM-based Seq2Seq (`glstm`)

The Long Short-Term Memory (LSTM) network is a type of Recurrent Neural Network (RNN). It processes sequences token-by-token, maintaining an internal "memory" or hidden state that captures information from previous steps.

*   **How it Works**: An Encoder LSTM reads the entire grapheme sequence and compresses its meaning into a fixed-size context vector (the final hidden and cell states). A Decoder LSTM then uses this context vector to generate the phoneme sequence one phoneme at a time, using its own previously generated phoneme as input for the next step.

*   **Why use it for G2P?**
    *   **Efficiency**: LSTMs are generally less parameter-heavy than Transformers and can train effectively on smaller datasets and with less computational hardware.
    *   **Sequential Intuition**: The step-by-step nature of RNNs is a very natural fit for sequence generation tasks like language and phoneme prediction.
    *   **Strong Baseline**: They are a proven technology and serve as a robust baseline to measure the benefits of more complex architectures.

*   **Limitations**:
    *   **Information Bottleneck**: All information from the input sequence must be crammed into one fixed-size context vector, which can be a problem for long or complex words.
    *   **Sequential Nature**: It cannot be parallelized during training, as each step `t` depends on the completion of step `t-1`.

<!-- You can add a diagram of the LSTM Encoder-Decoder model here -->

### Transformer-based Seq2Seq (`transformer`)

The Transformer architecture eschews recurrence and relies entirely on **self-attention mechanisms** to draw global dependencies between input and output.

*   **How it Works**: Instead of sequential processing, the self-attention mechanism allows the model to look at all other tokens in the sequence simultaneously when encoding a specific token. This gives it a complete, parallel view of the input. The decoder uses a similar mechanism, attending to both the source sequence and the phonemes it has generated so far.

*   **Why use it for G2P?**
    *   **Superior Context Handling**: The attention mechanism excels at modeling long-range dependencies. For example, it can easily learn that a final 'e' in a word like "bake" influences the pronunciation of the distant 'a'.
    *   **Parallelization**: Since it doesn't process tokens one-by-one, the computations can be heavily parallelized, leading to significantly faster training times on modern hardware (GPUs/TPUs).
    *   **State-of-the-Art Performance**: Transformers are the foundation for nearly all modern state-of-the-art models in NLP and have demonstrated superior performance on a wide range of tasks.

*   **Limitations**:
    *   **Data-Hungry**: They often require larger datasets to learn the complex relationships that LSTMs might handle with inductive bias.
    *   **Computational Cost**: While training can be faster, the model itself is typically larger and more complex, with a higher memory footprint.

<!-- You can add a diagram of the Transformer model here -->

## Motivation and Project Context

While several G2P tools exist, this project was created to fill a specific educational and practical gap.

*   **Existing Projects**: Standard tools like **Phonetisaurus** (based on Weighted Finite-State Transducers - WFSTs) and libraries like `g2p-en` are powerful but can have complex dependencies or present a steeper learning curve for those new to the field. Many are also not implemented in modern deep learning frameworks.

*   **Why This Project?**
    1.  **Direct Comparison**: To provide an apples-to-apples test bed for comparing LSTM and Transformer architectures on the exact same dataset and evaluation pipeline.
    2.  **Modern Framework**: To offer a G2P solution built entirely in **PyTorch**, a popular and flexible framework for deep learning research and deployment.
    3.  **Clarity and Simplicity**: To create a project with minimal dependencies and a focus on readable, well-commented code that is easy to follow, modify, and integrate into larger speech synthesis pipelines.

## Getting Started

Each model is in its own self-contained directory. Navigate to the desired folder to begin.

```bash
# Choose your preferred architecture
cd lstm/
# OR
# cd g2p-transformer/
```

### 1. Prerequisites

Install the required Python packages. It's recommended to use a virtual environment.
(this project uses 3.12 Python version)

```bash
# upgrade pip to lastest version
python -m pip install -U pip
```

```bash
# inside each project, for example lstm or transformer
# exists a requirements.txt
# some modules (or python libs) are specific for:
# - training
# - inference
# - benchmark
# In your project, install only minimal necessary to run
# theres
# requirements.txt - all packages
# requirements.train.txt - training/inference packages
# requirements.benchmarl.txt - inference/benchmark pachages
# requirements.inference.txt - inference only pachages, minimal
pip install -r requirements.txt
```

# README in progress...

### 2. Training a Model

From within the model's directory, run the training script. Your data should be in a `.tsv` file with `grapheme<TAB>phoneme` per line.

```bash
# To train the LSTM model
python G2P_train.py

# To train the Transformer model
python G2P_train_transformer.py
```

The best model checkpoint (`.pth`) and a training log will be saved.

### 3. Inference and Evaluation

Use the inference script to evaluate performance on a test set or convert new words.

```bash
# For the LSTM model
python G2P_inference.py --model_path g2p_model.pth --data_path pt_br.tsv

# For the Transformer model
python G2P_inference_transformer.py --model_path g2p_model_transformer.pth --data_path pt_br.tsv
```

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.