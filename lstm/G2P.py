import os

import torch
import torch.nn as nn
import random

CPU_THREADS = os.cpu_count()
CPU_THREADS_POWER = 0.9
__cpu_threads_total = int(CPU_THREADS * CPU_THREADS_POWER)
torch.set_num_threads(__cpu_threads_total)
#torch.set_num_interop_threads(__cpu_threads_total//4)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, bidirectional: bool = True):

        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * n_directions]
        # hidden, cell = [n_layers * n_directions, batch_size, hid_dim]
        if self.bidirectional:
            # Reshape hidden/cell from [n_layers * 2, batch, hid_dim] to [n_layers, 2, batch, hid_dim]
            hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
            cell = cell.view(self.n_layers, 2, -1, self.hid_dim)

            # Concatenate the forward (hidden[:, 0]) and backward (hidden[:, 1]) states
            hidden_cat = torch.cat((hidden[:, 0, ...], hidden[:, 1, ...]), dim=2)
            cell_cat = torch.cat((cell[:, 0, ...], cell[:, 1, ...]), dim=2)

            # Pass through the fully connected layer with tanh activation
            hidden_transformed = torch.tanh(self.fc(hidden_cat))
            cell_transformed = torch.tanh(self.fc(cell_cat))

            return hidden_transformed, cell_transformed
        else:
            # If not bidirectional, the shapes are already correct for the decoder
            return hidden, cell



class Decoder(nn.Module):

    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float,
                 pretrained_embeddings: torch.Tensor = None):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape[1] != emb_dim:
                 raise ValueError("Pretrained embedding dimension does not match model's emb_dim.")
            # (fine-tuning)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            # Default behavior: learn embeddings from scratch
            self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # input = [batch_size] -> precisa ser [1, batch_size]
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [1, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        # O primeiro input para o decoder é o token <sos>
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

    def predict(self, src_tensor: torch.Tensor, src_vocab, trg_vocab, max_len: int = 50, with_segment: bool = True) -> str:
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src_tensor)

            trg_sos_idx = trg_vocab.stoi['<sos>']
            trg_eos_idx = trg_vocab.stoi['<eos>']

            trg_indexes = [trg_sos_idx]

            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)

                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)

                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)

                if pred_token == trg_eos_idx:
                    break

        trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
        segment = " " if with_segment else ""
        return segment.join(trg_tokens[1:-1])

