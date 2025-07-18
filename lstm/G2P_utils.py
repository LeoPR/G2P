# G2P_utils.py (Final Corrected Version)
import torch
from torch.utils.data import Dataset
from collections import Counter
import random
from collections import defaultdict
import numpy as np
import Levenshtein
import regex as re

# --- Tokenizers ---
SOURCE_TOKENIZER_REGEX = re.compile(r"<[a-z]{2,3}|-[a-z]{2,4}>:|>:|\X")


def tokenize_source_text(text: str) -> list:
    """Tokenizes source text with sub-word tags."""
    return SOURCE_TOKENIZER_REGEX.findall(text)


def tokenize_phoneme_text(text: str) -> list:
    """Tokenizes a space-segmented phoneme string."""
    return text.split()


# --- Advanced Utilities ---
# (get_phoneme_ngrams and create_stratified_split are unchanged and correct)
def get_phoneme_ngrams(ipa_string, n_values=(2, 3)):
    ngrams = set()
    phonemes = ipa_string.split()  # Use split for space-segmented strings
    for n in n_values:
        if len(phonemes) >= n:
            for i in range(len(phonemes) - n + 1):
                ngrams.add(" ".join(phonemes[i:i + n]))
    return ngrams


def create_stratified_split(data_pairs, val_split_ratio=0.2):
    print("Iniciando divisão estratificada inteligente...")
    phoneme_freq, ngram_freq = Counter(), Counter()
    for _, ipa in data_pairs:
        phoneme_freq.update(ipa.split())
        ngram_freq.update(get_phoneme_ngrams(ipa))
    critical_phonemes = {p for p, count in phoneme_freq.items() if count == 1}
    critical_ngrams = {ng for ng, count in ngram_freq.items() if count == 1}
    train_indices, stratify_candidates = [], []
    for i, (word, ipa) in enumerate(data_pairs):
        phonemes, ngrams = set(ipa.split()), get_phoneme_ngrams(ipa)
        if phonemes.intersection(critical_phonemes) or ngrams.intersection(critical_ngrams):
            train_indices.append(i);
            continue
        min_freq, rarest_phoneme = float('inf'), ''
        for p in phonemes:
            if phoneme_freq.get(p, 0) < min_freq: min_freq, rarest_phoneme = phoneme_freq.get(p, 0), p
        word_len = len(word)
        length_bin = 'curta' if word_len <= 4 else 'média' if word_len <= 9 else 'longa'
        stratify_candidates.append({'index': i, 'key': (rarest_phoneme, length_bin)})
    strata_groups = defaultdict(list)
    for item in stratify_candidates: strata_groups[item['key']].append(item['index'])
    val_indices = []
    for _, indices in strata_groups.items():
        random.shuffle(indices)
        if len(indices) <= 2: train_indices.extend(indices); continue
        n_val = max(1, int(round(len(indices) * val_split_ratio)))
        val_indices.extend(indices[:n_val]);
        train_indices.extend(indices[n_val:])
    print(f"Divisão concluída: {len(train_indices)} amostras de treino, {len(val_indices)} de validação.")
    random.shuffle(train_indices);
    random.shuffle(val_indices)
    return train_indices, val_indices


# (calculate_feature_distance is unchanged and correct)
def calculate_feature_distance(phoneme_tokens1: list, phoneme_tokens2: list, feature_map: dict,
                               ins_del_penalty: int = 4) -> tuple[int, list]:
    """
    Calculates a phonetically-aware distance between two lists of phoneme tokens.
    """
    # Use Levenshtein's editops directly on the lists of tokens.
    ops = Levenshtein.editops(phoneme_tokens1, phoneme_tokens2)

    total_distance, error_descriptions = 0, []

    for op_type, pos1, pos2 in ops:
        if op_type == 'replace':
            # Get the phonemes directly from the input lists using the provided indices.
            p1, p2 = phoneme_tokens1[pos1], phoneme_tokens2[pos2]
            vec1, vec2 = feature_map.get(p1), feature_map.get(p2)
            if vec1 and vec2:
                feature_diff = sum(el1 != el2 for el1, el2 in zip(vec1, vec2))
                total_distance += feature_diff
                if feature_diff > 0: error_descriptions.append(f"Sub: '{p1}'->'{p2}' (Diff:{feature_diff})")
            else:
                total_distance += ins_del_penalty;
                error_descriptions.append(f"Sub: '{p1}'->'{p2}' (Unknown)")
        elif op_type == 'insert':
            p2 = phoneme_tokens2[pos2];
            total_distance += ins_del_penalty;
            error_descriptions.append(f"Ins: '{p2}'")
        elif op_type == 'delete':
            p1 = phoneme_tokens1[pos1];
            total_distance += ins_del_penalty;
            error_descriptions.append(f"Del: '{p1}'")

    return total_distance, error_descriptions


# --- Core Data Utilities ---
# (Vocabulary, G2PDataset, PadCollate, set_reproducibility are unchanged and correct)
class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list, tokenizer):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for token in tokenizer(sentence): frequencies[token] += 1
        for char, count in frequencies.items():
            if count >= self.freq_threshold: self.stoi[char], self.itos[idx] = idx, char; idx += 1

    def numericalize(self, text, tokenizer):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenizer(text)]


class G2PDataset(Dataset):
    def __init__(self, data_pairs, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
        self.data_pairs, self.src_vocab, self.trg_vocab = data_pairs, src_vocab, trg_vocab
        self.src_tokenizer, self.trg_tokenizer = src_tokenizer, trg_tokenizer

    def __len__(self): return len(self.data_pairs)

    def __getitem__(self, index):
        src, trg = self.data_pairs[index]
        src_numerical = [self.src_vocab.stoi["<sos>"]] + self.src_vocab.numericalize(src, self.src_tokenizer) + [
            self.src_vocab.stoi["<eos>"]]
        trg_numerical = [self.trg_vocab.stoi["<sos>"]] + self.trg_vocab.numericalize(trg, self.trg_tokenizer) + [
            self.trg_vocab.stoi["<eos>"]]
        return torch.tensor(src_numerical), torch.tensor(trg_numerical)


class PadCollate:
    def __init__(self, pad_idx): self.pad_idx = pad_idx

    def __call__(self, batch):
        srcs = [item[0] for item in batch];
        trgs = [item[1] for item in batch]
        srcs_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=False, padding_value=self.pad_idx)
        trgs_padded = torch.nn.utils.rnn.pad_sequence(trgs, batch_first=False, padding_value=self.pad_idx)
        return srcs_padded, trgs_padded


def set_reproducibility(seed, deterministic=True):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        print(
            "Running in deterministic mode."); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    else:
        print(
            "Running in non-deterministic mode."); torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True


# --- [MODIFIED] Expanded Phonetic Feature Map ---

# This version is compatible with the rich output from WikiPron.
# It includes entries for structural tokens ('.', '(', ')', '_'), modifier tokens ('ː'),
# and multi-character phonemes like affricates ('tʃ').
# To ensure all vectors have the same dimension, we add a 7th category to 'Manner' for 'Affricate'.
# All vectors are now 25 dimensions.

PHONETIC_FEATURE_MAP = {
    # Consonants (Plosives) - Manner[0]
    'p': [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'b': [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    't': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'd': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'g': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Consonants (Fricatives) - Manner[1]
    'f': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'v': [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    's': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'z': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ʃ': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ʒ': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ʁ': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'h': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'θ': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ð': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Consonants (Nasals) - Manner[2]
    'm': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'n': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ɲ': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ŋ': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Consonants (Liquids) - Manner[3:5]
    'l': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ʎ': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ɾ': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'w': [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'j': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Consonants (Affricates) - Manner[6]
    'tʃ': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'dʒ': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],

    # --- Vowels ---
    'i': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'e': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'ɛ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'a': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'u': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    'o': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
    'ɔ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    'æ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    'ɪ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'ʊ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
    'ə': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    'ʌ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    'ĩ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'ẽ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'ɐ̃': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    'ũ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    'õ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    'ɔɪ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    'aɪ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    'aʊ': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],

    # --- Structural and Modifier Tokens ---
    # Represented by zero vectors as they don't have traditional phonetic features.
    'ˈ': [0] * 25, 'ˌ': [0] * 25, '.': [0] * 25, '(': [0] * 25, ')': [0] * 25, 'ː': [0] * 25, '_': [0] * 25,
}


# --- [MODIFIED] Function using the new phonetic map ---

def create_phonetic_embedding_matrix(phonetic_map, vocab, embedding_dim):
    """
    Creates an embedding matrix initialized with phonetic features.
    """
    # [MODIFIED] Get feature_dim safely from a known phoneme.
    feature_dim = len(phonetic_map['p'])

    embedding_matrix = torch.randn(len(vocab), embedding_dim) * 0.01
    print(f"Creating phonetic embedding matrix. Feature dim: {feature_dim}, Target dim: {embedding_dim}")
    if feature_dim > embedding_dim:
        raise ValueError("Embedding dimension must be >= phonetic feature dimension.")

    for char, i in vocab.stoi.items():
        if char in phonetic_map:
            feature_vector = torch.tensor(phonetic_map[char], dtype=torch.float)
            embedding_matrix[i, :feature_dim] = feature_vector

    embedding_matrix = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)
    embedding_matrix[vocab.stoi['<pad>']] = 0.0
    return embedding_matrix