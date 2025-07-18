import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import sys
import glob
import matplotlib.pyplot as plt

# [MODIFIED] Import the new target tokenizer
from G2P import Encoder, Decoder, Seq2Seq
from G2P_utils import (
    Vocabulary, G2PDataset, PadCollate, tokenize_source_text, tokenize_phoneme_text,
    create_phonetic_embedding_matrix, set_reproducibility, PHONETIC_FEATURE_MAP, create_stratified_split
)


def manage_checkpoints(model_save_path, model_prefix, max_to_keep):
    checkpoints = sorted(glob.glob(os.path.join(model_save_path, f'{model_prefix}*.pth')))
    if len(checkpoints) > max_to_keep:
        for ckpt_to_delete in checkpoints[:-max_to_keep]:
            print(f"Recycling old checkpoint: {os.path.basename(ckpt_to_delete)}")
            os.remove(ckpt_to_delete)


def plot_and_save_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(12, 6));
    plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-');
    plt.plot(val_losses, label='Validation Loss', marker='x', linestyle='--')
    plt.title('Training and Validation Loss Over Epochs');
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.legend();
    plt.grid(True);
    plt.savefig(save_path);
    plt.close()
    print(f"Loss curve plot saved to '{save_path}'")


def load_data_from_directory(dict_path: str) -> tuple[list, list, list]:
    if not os.path.isdir(dict_path): print(f"Error: Dictionary path '{dict_path}' not found."); sys.exit(1)
    all_data_pairs, src_sentences, trg_sentences = [], [], []
    lang_files = {fn.replace('.tsv', '').replace('_', '-'): os.path.join(dict_path, fn) for fn in os.listdir(dict_path)
                  if fn.endswith('.tsv')}
    if not lang_files: print(f"Error: No .tsv dictionary files found in '{dict_path}'."); sys.exit(1)
    print(f"Found dictionaries: {list(lang_files.keys())}")
    for lang_tag, data_path in lang_files.items():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, ipa = parts
                    all_data_pairs.append((word, ipa));
                    src_sentences.append(f'<{lang_tag}>:{word}');
                    trg_sentences.append(ipa)
    return all_data_pairs, src_sentences, trg_sentences


def train_fn(model, iterator, optimizer, criterion, clip, device):
    model.train();
    epoch_loss = 0
    for batch in tqdm(iterator, total=len(iterator), desc="Training"):
        src, trg = batch;
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad();
        output = model(src, trg)
        output_dim = output.shape[-1];
        output = output[1:].view(-1, output_dim);
        trg = trg[1:].view(-1)
        loss = criterion(output, trg);
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip);
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate_fn(model, iterator, criterion, device):
    model.eval();
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(iterator, total=len(iterator), desc="Evaluating"):
            src, trg = batch;
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1];
            output = output[1:].view(-1, output_dim);
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def main():

    CPU_THREADS = os.cpu_count();
    torch.set_num_threads(int(CPU_THREADS * 0.9))
    SEED = 42;
    set_reproducibility(SEED, deterministic=False)
    USE_PHONETIC_EMBEDDING = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DICT_PATH = './dicts'
    MODEL_SAVE_PATH = './models';
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    HPARAMS = {
        'model_type': 'lstm_bidirectional', 'bidirectional_encoder': True, 'num_epochs': 100,
        'batch_size': 128, 'learning_rate': 0.001, 'enc_emb_dim': 256, 'dec_emb_dim': 256,
        'hid_dim': 512, 'n_layers': 2, 'enc_dropout': 0.5, 'dec_dropout': 0.5, 'clip': 1.0,
        'val_split_ratio': 0.005, 'save_every_n_epochs': 10, 'max_periodic_checkpoints': 3
    }
    print(f"Using device: {DEVICE}\nModels will be saved to '{MODEL_SAVE_PATH}/'")

    # --- Data Loading and Processing ---
    all_data_pairs, src_sentences, trg_sentences = load_data_from_directory(DICT_PATH)
    src_vocab = Vocabulary();
    trg_vocab = Vocabulary()
    print("Building source vocabulary...")
    src_vocab.build_vocabulary(src_sentences, tokenizer=tokenize_source_text)
    if "<und>:" not in src_vocab.stoi:
        print("Adding default '<und>:' tag to source vocabulary.")
        new_idx = len(src_vocab);
        src_vocab.stoi["<und>:"] = new_idx;
        src_vocab.itos[new_idx] = "<und>:"

    print("Building target vocabulary...")
    trg_vocab.build_vocabulary(trg_sentences, tokenizer=tokenize_phoneme_text)

    formatted_pairs = list(zip(src_sentences, trg_sentences))
    full_dataset = G2PDataset(
        data_pairs=formatted_pairs,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        src_tokenizer=tokenize_source_text,
        trg_tokenizer=tokenize_phoneme_text
    )

    train_indices, val_indices = create_stratified_split(all_data_pairs, val_split_ratio=HPARAMS['val_split_ratio'])
    train_dataset = Subset(full_dataset, train_indices);
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Source Vocab Size: {len(src_vocab)}, Target Vocab Size: {len(trg_vocab)}")
    print(f"Training Set: {len(train_dataset)}, Validation Set: {len(val_dataset)}")
    pad_idx = trg_vocab.stoi["<pad>"];
    collate_fn = PadCollate(pad_idx)
    num_workers = 0 if os.name == 'nt' else os.cpu_count() // 2
    train_iterator = DataLoader(train_dataset, batch_size=HPARAMS['batch_size'], shuffle=True, collate_fn=collate_fn,
                                num_workers=num_workers)
    val_iterator = DataLoader(val_dataset, batch_size=HPARAMS['batch_size'], collate_fn=collate_fn,
                              num_workers=num_workers)

    # --- Model Construction ---
    decoder_pretrained_embeddings = None
    if USE_PHONETIC_EMBEDDING:
        print("Using PHONETICALLY-AWARE embeddings for the decoder.")
        decoder_pretrained_embeddings = create_phonetic_embedding_matrix(
            phonetic_map=PHONETIC_FEATURE_MAP, vocab=trg_vocab, embedding_dim=HPARAMS['dec_emb_dim']
        )
    encoder = Encoder(
        len(src_vocab), HPARAMS['enc_emb_dim'], HPARAMS['hid_dim'], HPARAMS['n_layers'],
        HPARAMS['enc_dropout'], bidirectional=HPARAMS.get('bidirectional_encoder', False)
    )
    decoder = Decoder(
        len(trg_vocab), HPARAMS['dec_emb_dim'], HPARAMS['hid_dim'], HPARAMS['n_layers'],
        HPARAMS['dec_dropout'], pretrained_embeddings=decoder_pretrained_embeddings
    )
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # --- Training Loop ---
    best_valid_loss = float('inf');
    train_losses, val_losses = [], []
    with open("training_log.txt", "w") as log_file:
        for epoch in range(1, HPARAMS['num_epochs'] + 1):
            train_loss = train_fn(model, train_iterator, optimizer, criterion, HPARAMS['clip'], DEVICE)
            valid_loss = evaluate_fn(model, val_iterator, criterion, DEVICE)
            train_losses.append(train_loss);
            val_losses.append(valid_loss)
            log_message = f'Epoch: {epoch:03} | Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f}'
            print(log_message);
            log_file.write(log_message + '\n');
            log_file.flush()
            checkpoint = {'model_state_dict': model.state_dict(), 'src_vocab_stoi': src_vocab.stoi,
                          'src_vocab_itos': src_vocab.itos, 'trg_vocab_stoi': trg_vocab.stoi,
                          'trg_vocab_itos': trg_vocab.itos, 'params': HPARAMS}
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_path = os.path.join(MODEL_SAVE_PATH, f"g2p_{HPARAMS['model_type']}_best.pth")
                torch.save(checkpoint, best_model_path)
                print(f"\n--> New best model saved to '{best_model_path}'")
            if epoch % HPARAMS['save_every_n_epochs'] == 0:
                periodic_filename = f"g2p_{HPARAMS['model_type']}_e{epoch:03}_vl{valid_loss:.4f}.pth"
                periodic_path = os.path.join(MODEL_SAVE_PATH, periodic_filename)
                torch.save(checkpoint, periodic_path)
                print(f"\n--> Periodic checkpoint saved to '{periodic_path}'")
                manage_checkpoints(MODEL_SAVE_PATH, model_prefix=f"g2p_{HPARAMS['model_type']}_e",
                                   max_to_keep=HPARAMS['max_periodic_checkpoints'])

    plot_and_save_loss_curve(train_losses, val_losses, 'training_plot.png')
    print("Training complete. Final models and plot saved.")


if __name__ == '__main__':
    main()