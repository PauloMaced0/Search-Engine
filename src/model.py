import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCNNCrossEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=128, kernel_sizes=[5,7,9], num_layers=3, dropout=0.3, pretrained_embeddings=None):
        super().__init__()

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Stack of convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            ks = kernel_sizes[i % len(kernel_sizes)]
            conv = nn.Conv1d(in_channels=embedding_dim if i == 0 else num_filters,
                             out_channels=num_filters,
                             kernel_size=ks,
                             padding=ks//2)  # same padding
            self.convs.append(conv)

        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, joint_input):
        # joint_input: (B, L)
        x = self.embedding(joint_input)         # (B, L, E)
        x = x.transpose(1, 2)                   # (B, E, L)

        # Apply stacked conv layers with residual connections
        for conv in self.convs:
            residual = x
            x = self.activation(conv(x))
            x = x + residual                     # residual connection

        # Pool over sequence
        x = self.pool(x).squeeze(-1)            # (B, num_filters)
        logits = self.fc(self.dropout(x)).squeeze(-1)
        return logits
