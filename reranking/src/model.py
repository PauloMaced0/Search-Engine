import torch
import torch.nn as nn

class CNNInteractionBasedModel(nn.Module):
    def __init__(self, vocab_size, pretrained_embeddings, embedding_dim=300, num_filters=32, kernel_size=3):
        super(CNNInteractionBasedModel, self).__init__()
        self.vocab_size = vocab_size

        # Embedding layer to map token IDs to dense vectors
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layer to process interaction matrix
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(kernel_size, kernel_size))

        # Activation function
        self.activation = nn.ReLU()

        # Pooling layer
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

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
        # print("question ids:", query)
        # print("document ids:", document)

        # Embed query and document
        query_embed = self.embedding(query)  # (batch_size, query_len, embedding_dim)
        document_embed = self.embedding(document)  # (batch_size, doc_len, embedding_dim)

        # Compute interaction matrix
        # Interaction matrix shape: (batch_size, query_len, doc_len)
        interaction_matrix = torch.matmul(query_embed, document_embed.transpose(1, 2))

        # Add channel dimension for convolution
        interaction_matrix = interaction_matrix.unsqueeze(1)  # (batch_size, 1, query_len, doc_len)

        # Apply convolution
        conv_out = self.conv(interaction_matrix)  # (batch_size, num_filters, new_h, new_w)

        # Apply activation
        conv_out = self.activation(conv_out)

        # Apply max poolin
        pooled_out = self.pool(conv_out).squeeze(-1).squeeze(-1)  # (batch_size, num_filters)

        # Linear layer to compute logits
        logits = self.fc(pooled_out).squeeze(-1)  # (batch_size,)

        # Apply sigmoid to get probabilities
        prob = self.sigmoid(logits)  # (batch_size,)

        return prob
