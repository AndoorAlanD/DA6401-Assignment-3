import os
import wandb
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from prettytable import PrettyTable
import matplotlib.font_manager as fm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['WANDB_API_KEY'] = 'wandb.ai key'
wandb.login()

START_token = 0
END_token = 1

class Vocab:
    def __init__(self, lang_name):
        self.lang_name = lang_name
        self.num_chars = 2  # START and END
        self.char_to_idx = {}
        self.char_freq = {}
        self.idx_to_char = {0: "0", 1: "1"}

    def add_text(self, text):
        for char in text:
            self._add_char(char)

    def _add_char(self, char):
        if char not in self.char_to_idx:
            self.char_to_idx[char] = self.num_chars
            self.char_freq[char] = 1
            self.idx_to_char[self.num_chars] = char
            self.num_chars += 1
        else:
            self.char_freq[char] += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_embed_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cell_type', type=str, default='lstm', choices=['rnn', 'gru', 'lstm'])
    return parser.parse_args()


def text_to_indices(vocab, text):
    return [vocab.char_to_idx[char] for char in text]

def tensor_from_text(vocab, text):
    indices = text_to_indices(vocab, text)
    indices.append(END_token)
    return torch.tensor(indices, dtype=torch.long, device=device).view(1, -1)

def indices_to_text(vocab, tensor):
    output = ""
    for idx in tensor:
        if idx.item() == END_token:
            break
        output += vocab.idx_to_char[idx.item()]
    return output

def prepare_dataloader(df, source_vocab, target_vocab, batch_sz):
    data_pairs = list(zip(df[1].values, df[0].values))
    num_samples = len(data_pairs)
    source_tensor = np.zeros((num_samples, MAX_SEQ_LEN), dtype=np.int32)
    target_tensor = np.zeros((num_samples, MAX_SEQ_LEN), dtype=np.int32)

    for i, (src, tgt) in enumerate(data_pairs):
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        source_vocab.add_text(src)
        target_vocab.add_text(tgt)
        src_idx = text_to_indices(source_vocab, src)
        tgt_idx = text_to_indices(target_vocab, tgt)
        src_idx.append(END_token)
        tgt_idx.append(END_token)
        source_tensor[i, :len(src_idx)] = src_idx
        target_tensor[i, :len(tgt_idx)] = tgt_idx

    mask = (source_tensor != 0).astype(np.uint8)
    mask_tensor = torch.BoolTensor(mask).to(device)
    dataset = TensorDataset(torch.LongTensor(source_tensor).to(device), torch.LongTensor(target_tensor).to(device), mask_tensor)
    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_sz)

class EncoderNet(nn.Module):
    def __init__(self, cfg, input_dim):
        super(EncoderNet, self).__init__()
        self.embed = nn.Embedding(input_dim, cfg.inp_embed_size)
        self.rnn = algorithms[cfg.cell_type](cfg.inp_embed_size, cfg.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        embedded = self.dropout(self.embed(x))
        return self.rnn(embedded)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, mask=None):
        score = self.V(torch.tanh(self.W(query) + self.U(keys))).squeeze(2).unsqueeze(1)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attn_weights, keys)
        return context, attn_weights

class DecoderNet(nn.Module):
    def __init__(self, cfg, output_dim):
        super(DecoderNet, self).__init__()
        self.embedding = nn.Embedding(output_dim, cfg.hidden_dim)
        self.attn = AttentionLayer(cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.rnn = algorithms[cfg.cell_type](cfg.hidden_dim * 2, cfg.hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(cfg.hidden_dim, output_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, enc_out, enc_hidden, target_seq=None, src_mask=None):
        bs = enc_out.size(0)
        dec_input = torch.full((bs, 1), START_token, dtype=torch.long, device=device)
        dec_hidden = enc_hidden
        predictions = []
        attention_weights = []

        for t in range(MAX_SEQ_LEN):
            pred, dec_hidden, attn = self._step(dec_input, dec_hidden, enc_out)
            predictions.append(pred)
            attention_weights.append(attn)
            if target_seq is not None:
                dec_input = target_seq[:, t].unsqueeze(1)
            else:
                dec_input = pred.argmax(2).detach()

        output = torch.cat(predictions, dim=1)
        return F.log_softmax(output, dim=-1), dec_hidden, torch.cat(attention_weights, dim=1)

    def _step(self, input_token, hidden, enc_outputs):
        emb = self.dropout(self.embedding(input_token))
        if isinstance(hidden, tuple):
            h = hidden[0]
        else:
            h = hidden
        query = h.permute(1, 0, 2)
        context, attn = self.attn(query, enc_outputs)
        rnn_input = torch.cat((emb, context), dim=2)
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        return self.fc_out(rnn_out), hidden, attn

def run_epoch(loader, enc, dec, enc_opt, dec_opt, loss_fn, bsz, use_teacher=True):
    total_loss, correct, total = 0, 0, 0

    for x, y, mask in loader:
        y_tf = y if use_teacher else None
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        enc_out, enc_hid = enc(x)
        dec_out, _, _ = dec(enc_out, enc_hid, y_tf, mask)

        pred = dec_out.view(-1, dec_out.size(-1))
        gold = y.view(-1)
        loss = loss_fn(pred, gold)
        loss.backward()
        enc_opt.step()
        dec_opt.step()

        total_loss += loss.item()
        pred_ids = pred.argmax(1)
        bs = y.size(0)
        for i in range(bs):
            s = i * y.size(1)
            e = s + y.size(1)
            if torch.equal(pred_ids[s:e], gold[s:e]):
                correct += 1
        total += bs

    return total_loss / len(loader), (correct * 100) / total

def train_model(train_dl, val_dl, test_dl, enc, dec, cfg, epochs=20):
    enc_opt = optim.Adam(enc.parameters(), lr=cfg.lr)
    dec_opt = optim.Adam(dec.parameters(), lr=cfg.lr)
    loss_fn = nn.NLLLoss()

    for ep in range(1, epochs + 1):
        print("Epoch:", ep)
        train_loss, train_acc = run_epoch(train_dl, enc, dec, enc_opt, dec_opt, loss_fn, cfg.batch_size)
        print("Train: Accuracy:", train_acc, "Loss:", train_loss)
        if train_acc < 0.01 and ep >= 15:
            break
        val_loss, val_acc = run_epoch(val_dl, enc, dec, enc_opt, dec_opt, loss_fn, cfg.batch_size, use_teacher=False)
        print("Val: Accuracy:", val_acc, "Loss:", val_loss)
        wandb.log({'train_accuracy': train_acc, 'train_loss': train_loss, 'val_accuracy': val_acc, 'val_loss': val_loss})

    test_loss, test_acc = run_epoch(test_dl, enc, dec, enc_opt, dec_opt, loss_fn, cfg.batch_size, use_teacher=False)
    print("Test: Accuracy:", test_acc, "Loss:", test_loss, "\n")




if __name__ == '__main__':
    MAX_SEQ_LEN = 50

    config = parse_args()
    wandb.init(project='DL_A3', config=vars(config))
    wandb.run.name = (f"{config.cell_type}-do_{config.dropout}-bs_{config.batch_size}-lr_{config.lr}-"
                      f"hd_{config.hidden_dim}-emb_{config.inp_embed_size}")
    
    eng_vocab = Vocab('english')
    mal_vocab = Vocab('malayalam')

    train_df = pd.read_csv('/kaggle/input/malayalam/ml/lexicons/ml.translit.sampled.train.tsv', sep='\t', header=None)
    val_df = pd.read_csv('/kaggle/input/malayalam/ml/lexicons/ml.translit.sampled.dev.tsv', sep='\t', header=None)
    test_df = pd.read_csv('/kaggle/input/malayalam/ml/lexicons/ml.translit.sampled.test.tsv', sep='\t', header=None)

    algorithms = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    train_dl = prepare_dataloader(train_df, eng_vocab, mal_vocab, config.batch_size)
    val_dl = prepare_dataloader(val_df, eng_vocab, mal_vocab, config.batch_size)
    test_dl = prepare_dataloader(test_df, eng_vocab, mal_vocab, config.batch_size)

    encoder_net = EncoderNet(config, eng_vocab.num_chars).to(device)
    decoder_net = DecoderNet(config, mal_vocab.num_chars).to(device)
    train_model(train_dl, val_dl, test_dl, encoder_net, decoder_net, config)

    wandb.finish()
