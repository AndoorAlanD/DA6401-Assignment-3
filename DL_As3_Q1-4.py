import os
import wandb
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['WANDB_API_KEY'] = 'wandb.ai key'
wandb.login()

SOW_TOKEN = 0
EOW_TOKEN = 1

class Vocabulary:
    def __init__(self, lang_name):
        self.name = lang_name
        self.n_chars = 2  # For SOW and EOW
        self.char2idx = {}
        self.char_counts = {}
        self.idx2char = {SOW_TOKEN: "0", EOW_TOKEN: "1"}

    def add_word(self, word):
        for ch in word:
            self.add_char(ch)

    def add_char(self, ch):
        if ch not in self.char2idx:
            self.char2idx[ch] = self.n_chars
            self.char_counts[ch] = 1
            self.idx2char[self.n_chars] = ch
            self.n_chars += 1
        else:
            self.char_counts[ch] += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_enc_layers', type=int, default=3)
    parser.add_argument('--num_dec_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cell_type', choices=['rnn', 'gru', 'lstm'], default='lstm')
    return parser.parse_args()

def word_to_indices(vocab, word):
    return [vocab.char2idx[ch] for ch in word]

def word_to_tensor(vocab, word):
    indices = word_to_indices(vocab, word)
    indices.append(EOW_TOKEN)
    return torch.tensor(indices, dtype=torch.long, device=device).view(1, -1)

def tensor_to_word(vocab, tensor):
    output = ""
    for idx in tensor:
        if idx.item() == EOW_TOKEN:
            break
        output += vocab.idx2char[idx.item()]
    return output

def prepare_dataloader(df, src_vocab, tgt_vocab, batch_sz):
    pairs = list(zip(df[1].values, df[0].values))
    n_samples = len(pairs)
    input_ids = np.zeros((n_samples, MAX_LENGTH), dtype=np.int32)
    output_ids = np.zeros((n_samples, MAX_LENGTH), dtype=np.int32)

    for i, (src, tgt) in enumerate(pairs):
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        src_vocab.add_word(src)
        tgt_vocab.add_word(tgt)
        src_idx = word_to_indices(src_vocab, src) + [EOW_TOKEN]
        tgt_idx = word_to_indices(tgt_vocab, tgt) + [EOW_TOKEN]
        input_ids[i, :len(src_idx)] = src_idx
        output_ids[i, :len(tgt_idx)] = tgt_idx

    dataset = TensorDataset(torch.LongTensor(input_ids).to(device),
                            torch.LongTensor(output_ids).to(device))
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_sz)

class Encoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.embed = nn.Embedding(input_dim, config.inp_embed_size)
        self.rnn = algorithms[config.cell_type](config.inp_embed_size, config.hidden_size, config.num_enc_layers,
                                                batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.dropout(self.embed(x))
        output, hidden = self.rnn(x)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.embed = nn.Embedding(output_dim, config.hidden_size)
        self.rnn = algorithms[config.cell_type](config.hidden_size, config.hidden_size, config.num_enc_layers,
                                                batch_first=True)
        self.fc = nn.Linear(config.hidden_size, output_dim)

    def forward(self, enc_outputs, enc_hidden, tgt_tensor=None):
        batch_sz = enc_outputs.size(0)
        input_token = torch.full((batch_sz, 1), SOW_TOKEN, dtype=torch.long, device=device)
        hidden = enc_hidden
        all_outputs = []

        for t in range(MAX_LENGTH):
            x = F.relu(self.embed(input_token))
            x, hidden = self.rnn(x, hidden)
            logits = self.fc(x)
            all_outputs.append(logits)

            if tgt_tensor is not None:
                input_token = tgt_tensor[:, t].unsqueeze(1)
            else:
                input_token = logits.argmax(-1).detach()

        all_outputs = torch.cat(all_outputs, dim=1)
        return F.log_softmax(all_outputs, dim=-1), hidden, None

def run_epoch(loader, encoder, decoder, opt_enc, opt_dec, criterion, batch_sz, train=True):
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = batch
        bsz = inputs.size(0)
        tgt_forced = targets if train else None

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        enc_out, enc_hidden = encoder(inputs)
        dec_out, _, _ = decoder(enc_out, enc_hidden, tgt_forced)

        logits = dec_out.view(-1, dec_out.size(-1))
        labels = targets.view(-1)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            opt_enc.step()
            opt_dec.step()

        epoch_loss += loss.item()

        preds = logits.argmax(1).view(bsz, MAX_LENGTH)
        labels = labels.view(bsz, MAX_LENGTH)
        correct += (preds == labels).all(dim=1).sum().item()
        total += bsz

    return epoch_loss / len(loader), (100 * correct) / total

def run_training(train_dl, val_dl, test_dl, encoder, decoder, config, n_epochs=2):
    opt_enc = optim.Adam(encoder.parameters(), lr=config.lr)
    opt_dec = optim.Adam(decoder.parameters(), lr=config.lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}")
        train_loss, train_acc = run_epoch(train_dl, encoder, decoder, opt_enc, opt_dec, criterion, config.batch_size, train=True)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        if train_acc < 1.0 and epoch >= 15:
            break

        val_loss, val_acc = run_epoch(val_dl, encoder, decoder, opt_enc, opt_dec, criterion, config.batch_size, train=False)
        print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        wandb.log({'train_loss': train_loss, 'train_accuracy': train_acc,
                   'val_loss': val_loss, 'val_accuracy': val_acc})

    test_loss, test_acc = run_epoch(test_dl, encoder, decoder, opt_enc, opt_dec, criterion, config.batch_size, train=False)
    print(f"Test  Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

# ---------------- Main Code ----------------

if __name__ == "__main__":
    MAX_LENGTH = 50
    args = parse_args()
    wandb.init(project='DL_A3_CLI', config=vars(args))
    config = wandb.config

    wandb.run.name = (
        f"{config.cell_type}-E_{config.num_enc_layers}-D_{config.num_dec_layers}-"
        f"do_{config.dropout}-bs_{config.batch_size}-lr_{config.lr}-"
        f"hs_{config.hidden_size}-emb_{config.inp_embed_size}-"
    )

    # Data preparation
    input_vocab = Vocabulary('English')
    output_vocab = Vocabulary('Malayalam')

    train_data = pd.read_csv('/kaggle/input/malayalam/ml/lexicons/ml.translit.sampled.train.tsv', sep='\t', header=None)
    val_data = pd.read_csv('/kaggle/input/malayalam/ml/lexicons/ml.translit.sampled.dev.tsv', sep='\t', header=None)
    test_data = pd.read_csv('/kaggle/input/malayalam/ml/lexicons/ml.translit.sampled.test.tsv', sep='\t', header=None)

    train_loader = prepare_dataloader(train_data, input_vocab, output_vocab, args.batch_size)
    val_loader = prepare_dataloader(val_data, input_vocab, output_vocab, args.batch_size)
    test_loader = prepare_dataloader(test_data, input_vocab, output_vocab, args.batch_size)

    algorithms = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    encoder = Encoder(args, input_vocab.n_chars).to(device)
    decoder = Decoder(args, output_vocab.n_chars).to(device)

    run_training(train_loader, val_loader, test_loader, encoder, decoder, args)
    wandb.finish()
