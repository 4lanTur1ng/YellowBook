import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                           bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # 这里hidden_dim乘以2是因为是双向，需要拼接两个方向，跟n_layers的层数无关。

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text.shape=[seq_len, batch_size]
        embedded = self.dropout(self.embedding(text))
        # output: [batch,seq,2*hidden if bidirection else hidden]
        # hidden/cell: [bidirec * n_layers, batch, hidden]
        output, (hidden, cell) = self.rnn(embedded)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]，

        return self.fc(hidden.squeeze(0))  # 在接一个全连接层，最终输出[batch size, output_dim]


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class RNN_ATTs(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0, hidden_size2=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
