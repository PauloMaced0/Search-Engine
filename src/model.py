import torch
import torch.nn as nn

class CNNInteractionBasedModel(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings=None, embedding_dim=300, num_filters=32, kernel_size=3, dropout=0.3):
        super().__init__()
        # Embedding layer to map token IDs to dense vectors
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layer to process interaction matrix
        self.conv = nn.Conv2d(1, out_channels=num_filters, kernel_size=(kernel_size, kernel_size))

        # Activation function
        self.activation = nn.ReLU()

        # Pooling layer
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        self.dropout = nn.Dropout(dropout)

        # Fully connected layer for scoring
        self.fc = nn.Linear(num_filters, 1)

        # Sigmoid for probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, document):
        """
        Args:
        - query (torch.Tensor): Tensor of query token IDs (batch_size, query_len)
        - document (torch.Tensor): Tensor of document token IDs (batch_size, doc_len)
        
        Returns:
        - prob (torch.Tensor): Relevance score (batch_size,)
        """
        # Embed 
        query_embed = self.embedding(query)  # (B, Q, E)
        document_embed = self.embedding(document)  # (B, D, E)

        # Compute interaction matrix
        # Interaction matrix shape: (B, Q, D)
        interaction_matrix = torch.matmul(query_embed, document_embed.transpose(1, 2))

        # Convolution
        conv_out = self.activation(self.conv(interaction_matrix.unsqueeze(1)))  # (B, F, h, w)
        pooled = self.pool(conv_out).squeeze(-1).squeeze(-1)  # (B, F)

        # Dropout + FC
        logits = self.fc(self.dropout(pooled)).squeeze(-1)  # (B,)

        return logits
