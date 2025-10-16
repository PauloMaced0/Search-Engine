import torch.nn as nn
from transformers import BertModel

class BertCrossEncoder(nn.Module):
    def __init__(self, vocab_size, pretrained_model="bert-base-uncased", dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tensor (B, L)
            attention_mask: Tensor (B, L)
        Returns:
            logits: Tensor (B,)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(self.dropout(cls_embedding)).squeeze(-1)
        return logits
