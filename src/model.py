import torch.nn as nn

class CNNCrossEncoder(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings=None, embedding_dim=300, num_filters=64, kernel_size=3, dropout=0.3):
        super().__init__()

        # Embedding layer (shared for query and doc)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 1D CNN over the concatenated sequence (query + doc)
        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=num_filters,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, joint_input):
        """
        Args:
            joint_input (Tensor): token IDs for [query + SEP + document], shape (B, L)
        Returns:
            logits (Tensor): relevance scores, shape (B,)
        """
        # (B, L, E)
        embed = self.embedding(joint_input)
        # CNN expects (B, E, L)
        x = embed.transpose(1, 2)

        # Apply convolution and pooling
        conv_out = self.activation(self.conv(x))
        pooled = self.pool(conv_out).squeeze(-1)

        logits = self.fc(self.dropout(pooled)).squeeze(-1)
        return logits
