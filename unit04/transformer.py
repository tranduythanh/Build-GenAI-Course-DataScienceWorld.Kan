import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Số attention head
        self.num_heads = num_heads
        # Số chiều của mô hình
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        # độ sâu của mô hình
        self.depth = d_model // num_heads

        # Các phép chiếu tuyến tính cho query, key, value
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Phép chiếu tuyến tính cuối cùng
        self.final_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Chia tensor thành nhiều head
        # Đầu vào: (batch_size, seq_len, d_model)
        # Đầu ra: (batch_size, num_heads, seq_len, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Tạo query, key, value từ đầu vào
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # Chia thành nhiều head
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Tính scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Chia cho căn bậc hai của độ sâu
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        # Áp dụng mask nếu có
        if mask is not None:
            # Đảm bảo mask có kích thước phù hợp với scaled_attention_logits
            if mask.dim() == 2:
                # Nếu mask có kích thước [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            elif mask.dim() == 3 and mask.size(1) == 1:
                # Nếu mask có kích thước [batch_size, 1, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]

            # Mở rộng mask để phù hợp với kích thước của scaled_attention_logits
            if mask.size(-1) != scaled_attention_logits.size(-1):
                # Nếu độ dài của mask khác với độ dài của seq_len_k
                # Cắt hoặc mở rộng mask
                if mask.size(-1) < scaled_attention_logits.size(-1):
                    # Mở rộng mask
                    padding = torch.ones((mask.size(0), mask.size(1), mask.size(2),
                                         scaled_attention_logits.size(-1) - mask.size(-1)),
                                         device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([mask, padding], dim=-1)
                else:
                    # Cắt mask
                    mask = mask[..., :scaled_attention_logits.size(-1)]

            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

        # Áp dụng softmax để tính trọng số chú ý
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Tính đầu ra
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)

        # Ghép các head lại với nhau
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        # Áp dụng phép chiếu tuyến tính cuối cùng
        output = self.final_linear(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        # Mạng feed-forward hai lớp
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Sử dụng hàm kích hoạt GELU (phổ biến trong BERT)
        return self.linear2(F.gelu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoding, self).__init__()

        # Tạo encoding cho mỗi vị trí
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Áp dụng hàm sin cho chỉ số chẵn và cos cho chỉ số lẻ
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Đăng ký buffer (không phải là tham số nhưng là một phần của trạng thái module)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Thêm positional encoding vào input
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff)

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Multi-head attention (self-attention)
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # Add & Norm

        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # Add & Norm

        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        # Multi-head attention cho self-attention (với mask)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # Multi-head attention cho encoder-decoder attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff)

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # Masked multi-head attention (self-attention)
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        # Multi-head attention (encoder-decoder attention)
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        # Feed forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length=512, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Lớp nhúng từ vựng
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Mã hóa vị trí
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Các lớp encoder
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # x là token IDs đầu vào: (batch_size, seq_len)

        # Chuyển token IDs thành embedding và thêm mã hóa vị trí
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        x = self.dropout(x)

        # Đi qua từng lớp encoder
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, seq_len, d_model)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length=512, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Lớp embedding từ vựng
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Mã hóa vị trí
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Các lớp decoder
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # x là token IDs đầu vào: (batch_size, seq_len)

        # Chuyển token IDs thành embedding và thêm mã hóa vị trí
        x = self.embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        x = self.dropout(x)

        # Đi qua từng lớp decoder
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

        return x  # (batch_size, seq_len, d_model)

class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 enc_layers=6, dec_layers=6, max_seq_length=512, dropout_rate=0.1):
        super(TranslationModel, self).__init__()

        # Encoder từ mô hình BERT
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_heads,
            d_ff,
            enc_layers,
            max_seq_length,
            dropout_rate
        )

        # Decoder
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            d_ff,
            dec_layers,
            max_seq_length,
            dropout_rate
        )

        # Lớp tuyến tính cuối cùng để dự đoán từ tiếp theo
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)

    def create_masks(self, src, tgt):
        # Tạo mask cho padding của câu nguồn
        src_padding_mask = (src == 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)

        # Tạo mask cho padding của câu đích
        tgt_padding_mask = (tgt == 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)

        # Tạo look-ahead mask cho decoder (để decoder không thấy được các từ trong tương lai)
        tgt_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones(1, tgt_len, tgt_len), diagonal=1).type_as(tgt)
        look_ahead_mask = look_ahead_mask == 1  # Chuyển sang boolean

        # Kết hợp look-ahead mask với padding mask
        combined_mask = torch.max(tgt_padding_mask, look_ahead_mask)

        return src_padding_mask, combined_mask

    def forward(self, src, tgt):
        # Tạo các mask
        src_padding_mask, combined_mask = self.create_masks(src, tgt)

        # Đi qua encoder
        enc_output = self.encoder(src, src_padding_mask)

        # Đi qua decoder
        dec_output = self.decoder(tgt, enc_output, combined_mask, src_padding_mask)

        # Đi qua lớp tuyến tính cuối cùng và trả về logits
        logits = self.final_layer(dec_output)

        return logits