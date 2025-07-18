# G2P_inference_minimal.py
# ------------------------------

import torch
import os
import time
import sys

from G2P_inference import G2PInference

if __name__ == '__main__':
    # --- Configuration ---
    # Set to a reasonable number to not hog the system
    if os.cpu_count():
        torch.set_num_threads(min(4, os.cpu_count()))

    # --- DEMONSTRATION ---
    # This path should point to the model saved by G2P_train.py
    MODEL_PATH = './models/g2p_lstm_bidirectional_best.pth'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # You must provide a language tag that the model was trained on
    LANG_TAG_FOR_TESTING = 'pt-br'
    WORDS_TO_TEST = ['fantabulástico', 'computador', 'hylbert', 'xícara', 'psicologia', 'grapheme']

    # --- 1. Instantiate the G2P class ---
    init_start_time = time.time()
    try:
        # The G2PInference class is now imported from G2P_inference.py
        g2p = G2PInference(MODEL_PATH, device=DEVICE)
    except FileNotFoundError:
        print(f"\n[ERROR] Model file not found at '{MODEL_PATH}'.")
        print("Please ensure you have run G2P_train.py to generate the model file,")
        print("or that the path is correct.")
        sys.exit(1)

    print(f"G2P class initialized in: {time.time() - init_start_time:.4f}s")
    print("-" * 40)

    # --- 2. Use the .predict() method ---
    print(f"Performing inference for language '{LANG_TAG_FOR_TESTING}'...\n")

    inference_start_time = time.time()
    for word in WORDS_TO_TEST:
        # The .predict() method comes from the imported class
        ipa_sound = g2p.predict(word, lang_tag=LANG_TAG_FOR_TESTING)
        # We can join the phonemes for a cleaner display
        ipa_display = ''.join(ipa_sound.split())
        print(f"G2P('{word}') -> '{ipa_display}'")

    total_time = time.time() - inference_start_time
    avg_time_ms = (total_time / len(WORDS_TO_TEST)) * 1000

    print("\n" + "-" * 40)
    print(f"Total inference time for {len(WORDS_TO_TEST)} words: {total_time:.4f}s")
    print(f"Average time per word: {avg_time_ms:.2f} ms")