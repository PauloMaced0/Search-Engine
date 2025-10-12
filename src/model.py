import torch.nn as nn

class DeepCNNCrossEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=128, kernel_sizes=[5, 7, 9], num_layers=3, dropout=0.3, pretrained_embeddings=None):
        super().__init__()

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layers
        self.convs = nn.ModuleList()
        self.projs = nn.ModuleList()

        for i in range(num_layers):
            in_ch = embedding_dim if i == 0 else num_filters
            ks = kernel_sizes[i % len(kernel_sizes)]

            conv = nn.Conv1d(in_channels=in_ch,
                             out_channels=num_filters,
                             kernel_size=ks,
                             padding=ks // 2)
            self.convs.append(conv)

            # projection only if dimensions differ
            if in_ch != num_filters:
                self.projs.append(nn.Conv1d(in_ch, num_filters, kernel_size=1))
            else:
                self.projs.append(nn.Identity())

        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, joint_input):
        x = self.embedding(joint_input)  # (B, L, E)
        x = x.transpose(1, 2)            # (B, E, L)

        # Residual stack
        for conv, proj in zip(self.convs, self.projs):
            residual = proj(x)
            x = self.activation(conv(x))
            x = x + residual

        x = self.pool(x).squeeze(-1)
        logits = self.fc(self.dropout(x)).squeeze(-1)
        return logits