"""
新闻文本分类 — 基于 BERT 架构的文本分类器

模型结构 (BERT 风格):
  Input: [CLS] + word_ids (batch, seq_len)
  ├── Token Embedding(vocab_size, d_model)
  ├── Learned Position Embedding(max_len, d_model)
  ├── Token Type Embedding(2, d_model) — 单句分类全为 0
  ├── LayerNorm + Dropout (BERT 标准 embedding 后处理)
  ├── TransformerEncoder (Post-LN, gelu) × num_layers
  ├── [CLS] Pooling — 取首 token 隐向量
  └── Classifier: Dropout → Linear(d_model, num_classes)

设计说明:
  - 使用 [CLS] token 做分类，是 BERT 的标志性设计
  - Post-LN: LayerNorm 在残差连接之后，与原版 BERT 一致
  - Token Type Embedding 保留以展示 BERT 完整输入结构（单句分类全为 segment 0）
"""
import math
import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    """BERT 风格的三合一 Embedding 层

    BERT 的输入表示 = Token Embedding + Position Embedding + Segment Embedding
    对单句分类任务，所有 token 的 segment id = 0
    """

    def __init__(self, vocab_size, d_model, max_len=1024, dropout=0.3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 0=句子A, 1=句子B
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, input_ids, token_type_ids=None):
        # input_ids: (batch, seq_len)
        batch_size, seq_len = input_ids.shape

        # Token Embedding
        token_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Position Embedding (learned, 和原版 BERT 一致)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Token Type (Segment) Embedding
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
        seg_emb = self.segment_embedding(token_type_ids)

        # BERT: 三路相加 → LayerNorm → Dropout
        embeddings = token_emb + pos_emb + seg_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NewsTextClassifier(nn.Module):
    """BERT 风格的新闻文本分类器

    与 HuggingFace BertForSequenceClassification 结构对齐:
      Embedding → TransformerEncoder (Post-LN) → [CLS] Pooling → Classifier
    """

    def __init__(self, vocab_size, num_classes=14, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=1024, max_len=1024, dropout=0.3):
        super().__init__()
        self.d_model = d_model

        self.embeddings = BertEmbeddings(vocab_size, d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=False  # Post-LN，与 BERT 原版一致
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embeddings.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.embeddings.position_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.embeddings.segment_embedding.weight, mean=0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len), 首列为 [CLS] token
        x = self.embeddings(input_ids)

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # [CLS] Pooling: 取第一个 token 的隐向量做分类 (BERT 标志性设计)
        cls_hidden = x[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)
        return logits
