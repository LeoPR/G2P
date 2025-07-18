
import torch
import argparse
import random
import Levenshtein
from jiwer import cer
from tqdm import tqdm
import os
import sys
CPU_THREADS = os.cpu_count()
CPU_THREADS_POWER = 0.9
__cpu_threads_total = int(CPU_THREADS * CPU_THREADS_POWER)
torch.set_num_threads(__cpu_threads_total)


from G2P import Encoder, Decoder, Seq2Seq
from G2P_utils import Vocabulary, tokenize_source_text, PHONETIC_FEATURE_MAP, calculate_feature_distance


class G2PInference:
    """A wrapper class for easy G2P model loading and prediction."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.src_vocab, self.trg_vocab = Vocabulary(), Vocabulary()
        self.src_vocab.stoi, self.src_vocab.itos = checkpoint['src_vocab_stoi'], checkpoint['src_vocab_itos']
        self.trg_vocab.stoi, self.trg_vocab.itos = checkpoint['trg_vocab_stoi'], checkpoint['trg_vocab_itos']

        params = checkpoint['params']
        params['input_dim'], params['output_dim'] = len(self.src_vocab), len(self.trg_vocab)

        encoder = Encoder(
            params['input_dim'], params['enc_emb_dim'], params['hid_dim'], params['n_layers'],
            params['enc_dropout'], bidirectional=params.get('bidirectional_encoder', True)
        )
        decoder = Decoder(
            params['output_dim'], params['dec_emb_dim'], params['hid_dim'], params['n_layers'],
            params['dec_dropout']
        )
        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded successfully.")

    def predict(self, word: str, lang_tag: str = None) -> str:
        self.model.eval()
        effective_tag = lang_tag if lang_tag is not None else "und"
        source_string = f'<{effective_tag}>:{word.lower()}'
        tokens = ['<sos>'] + tokenize_source_text(source_string) + ['<eos>']
        numericalized_tokens = [self.src_vocab.stoi.get(t, self.src_vocab.stoi['<unk>']) for t in tokens]
        src_tensor = torch.LongTensor(numericalized_tokens).unsqueeze(1).to(self.device)

        phonetic_translation = self.model.predict(src_tensor, self.src_vocab, self.trg_vocab)
        return phonetic_translation

def evaluate_performance(g2p_instance: G2PInference, data_pairs: list, lang_tag: str, num_examples_to_show: int = 5):
    all_refs, all_hyps = [], []
    perfect_samples, good_samples, bad_samples = [], [], []
    correct_predictions = 0

    print(f"\nEvaluating {len(data_pairs)} examples for language '{lang_tag}'...")
    for word, true_ipa_str in tqdm(data_pairs, desc="Evaluating"):
        pred_ipa_str = g2p_instance.predict(word, lang_tag=lang_tag)

        true_ipa_tokens = true_ipa_str.split()
        pred_ipa_tokens = pred_ipa_str.split()

        # For PER, jiwer expects space-separated strings, so we can pass the originals.
        all_refs.append(true_ipa_str)
        all_hyps.append(pred_ipa_str)

        lev_distance = Levenshtein.distance(pred_ipa_tokens, true_ipa_tokens)
        feature_dist, errors = calculate_feature_distance(pred_ipa_tokens, true_ipa_tokens, PHONETIC_FEATURE_MAP)

        sample = f'Word: "{word}" | True: "{''.join(true_ipa_str.split())}" -> Pred: "{''.join(pred_ipa_str.split())}"'

        if lev_distance == 0:
            correct_predictions += 1
            if len(perfect_samples) < num_examples_to_show: perfect_samples.append(sample)
        elif feature_dist <= 2:
            if len(good_samples) < num_examples_to_show:
                good_samples.append(f"{sample} (Feat. Dist: {feature_dist}, Errors: {errors})")
        else:
            if len(bad_samples) < num_examples_to_show:
                bad_samples.append(f"{sample} (Feat. Dist: {feature_dist}, Errors: {errors})")

    phoneme_error_rate = cer(all_refs, all_hyps) * 100
    word_accuracy = (correct_predictions / len(data_pairs)) * 100 if data_pairs else 0

    print("\n" + "=" * 20 + " GENERAL METRICS " + "=" * 20)
    print(f"Word Accuracy (WA): {word_accuracy:.2f}% ({correct_predictions}/{len(data_pairs)} correct)")
    print(f"Phoneme Error Rate (PER): {phoneme_error_rate:.2f}%")
    print("\n" + "=" * 20 + " QUALITATIVE ANALYSIS " + "=" * 20)
    print(f"\n--- ✅ Perfect Predictions ---");
    print("\n".join(perfect_samples) if perfect_samples else "None found.")
    print(f"\n--- 👍 Good Predictions (small feature distance) ---");
    print("\n".join(good_samples) if good_samples else "None found.")
    print(f"\n--- ❌ Bad Predictions (large feature distance) ---");
    print("\n".join(bad_samples) if bad_samples else "None found.")
    print("-" * 58)



if __name__ == '__main__':
    CPU_THREADS = os.cpu_count();
    torch.set_num_threads(int(CPU_THREADS * 0.9))
    parser = argparse.ArgumentParser(description="G2P Model Inference and Evaluation")
    parser.add_argument("--model_path", type=str, default="./models/g2p_lstm_bidirectional_best.pth",
                        help="Path to the trained model (.pth).")
    parser.add_argument("--data_path", type=str, default="./dicts/pt-br.tsv", help="Path to the .tsv evaluation data.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu' or 'cuda').")
    parser.add_argument("--num_eval_samples", type=int,
                        #default=200000,
                        default=200,
                        help="Number of samples for performance evaluation.")
    parser.add_argument("--lang", type=str, default="pt-br", help="Language tag for the data (e.g., 'en-us', 'pt-br').")
    args = parser.parse_args()

    try:
        g2p = G2PInference(args.model_path, device=args.device)
    except FileNotFoundError:
        print(f"\nError: Model file not found at '{args.model_path}'");
        sys.exit(1)

    print("\n" + "=" * 30 + "\nPERFORMANCE EVALUATION\n" + "=" * 30)
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            all_data_pairs = [line.strip().split('\t') for line in f if '\t' in line]
        eval_data = random.sample(all_data_pairs, min(len(all_data_pairs), args.num_eval_samples))
        evaluate_performance(g2p, eval_data, lang_tag=args.lang)
    except FileNotFoundError:
        print(f"\nWarning: Evaluation data file not found at '{args.data_path}'. Skipping performance evaluation.")

    print("\n" + "=" * 30 + "\nSIMPLE API USAGE EXAMPLE\n" + "=" * 30)
    print(f"\n--- Predictions with specified tag '{args.lang}' ---")
    test_words_tagged = ["casa", 'casal', 'casinha', "bananão",
                         'psicologia',
                         'psicologias',
                         'chover',
                         'chovi',
                         'chovemos',
                         'chovido',
                         'miau',
                         'uau',
                         "inteligência",
                         "fonema", "hylbert", "joão pedro", "New York"]
    for word in test_words_tagged:
        ipa_result = g2p.predict(word, lang_tag=args.lang)
        print(f'"{word}" ({args.lang}) -> "{''.join(ipa_result.split())}"')

    print(f"\n--- Predictions with no tag (defaults to '<und>:') ---")
    test_words_untagged = ["phoneme", "grapheme", "default", "transformer"]
    for word in test_words_untagged:
        ipa_result = g2p.predict(word)
        print(f'"{word}" (default) -> "{''.join(ipa_result.split())}"')
