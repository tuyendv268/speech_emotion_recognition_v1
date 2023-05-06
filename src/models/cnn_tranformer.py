from torch.nn.utils import weight_norm
import torch.nn.functional as F 
from torch import nn
import torch
import math

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
# class PositionwiseFeedForward(nn.Module):

#     def __init__(self, d_model, hidden, drop_prob=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, hidden)
#         self.linear2 = nn.Linear(hidden, d_model)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=drop_prob)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        # output = self.layer_norm(output + residual)

        return output

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        
        self.d_k = d_model
        self.d_v = d_model
        
        self.W_Q = nn.Linear(d_model, d_model * num_attention_heads)
        self.W_K = nn.Linear(d_model, d_model * num_attention_heads)
        self.W_V = nn.Linear(d_model, d_model * num_attention_heads)
        
        self.ffw = nn.Linear(d_model * num_attention_heads, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, attention_masks, K=None, V=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(1, 2)
        
        attention_masks = attention_masks.unsqueeze(1).expand(batch_size, Q.size(1), Q.size(1))
        attention_masks = attention_masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
        
        context, attention_weights = ScaleDotProductAttention()(
            q=q_s, k=k_s, v=v_s,
            mask=attention_masks,
        )
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, 
            self.num_attention_heads * self.d_v)
        context = self.ffw(context)
        return context

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(
            d_model, n_head, drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.ffw = PositionwiseFeedForward(d_in=d_model, d_hid=ffn_hidden, kernel_size=(5, 1), dropout=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(Q=x, K=x, V=x, attention_masks=s_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x_ = x
        x = self.ffw(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x_)
        
        # x = self.sigmoid(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.d_model = d_model
        if d_model % 2 == 1:
            d_model += 1
            
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :self.d_model]
        return self.dropout(x)

class Transformer_Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Transformer_Encoder, self).__init__()
        
        self.position_embedding = PositionalEncoding(
            d_model=d_model
        )
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        

    def forward(self, features, attention_masks, device="cuda"):
        inputs = features
        
        if attention_masks is None:
            attention_masks = torch.ones(inputs.shape[0:2]).to(device)
        
        inputs = self.position_embedding(inputs)
        
        hiddens = []
        for layer in self.layers:
            inputs = layer(inputs, attention_masks)
            
            tmp = inputs.mean(dim=1).unsqueeze(1)
            hiddens.append(tmp)
            
        hiddens = torch.cat(hiddens, dim=1)
        return inputs, hiddens

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        channels = [39, 256, 39]
        kernels = [3, 3, 3]
        convs = []
        
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            conv = [
                weight_norm(nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernels[i],
                    padding="same",
                    stride=1)),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool1d(
                    kernel_size=2, 
                    stride=2)
            ]            
            convs += conv
        
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(0.2)

    
    def forward(self, inputs):
        for conv in self.convs:
            inputs = conv(inputs)
        inputs = self.dropout(inputs)
        return inputs

class CNN_Transformer(nn.Module):
    def __init__(self, config=None) -> None:
        super(CNN_Transformer, self).__init__()
        # d_model, ffn_hidden, n_head, n_layers, drop_prob
        self.cnn = CNN()
        self.weighted_layers = nn.Parameter(torch.randn(1, 8))
        self.transformers = Transformer_Encoder(
            d_model=39, 
            ffn_hidden=256, 
            n_head=6, 
            n_layers=4, 
            drop_prob=0.1)
        self.cls_head = nn.Linear(39, 8)
        
        
    def forward(self, inputs, lengths):
        inputs = self.cnn(inputs)
        inputs = inputs.transpose(1, 2)
        inputs, hiddens = self.transformers(inputs, None)

        outputs = torch.matmul(self.weighted_layers, hiddens)
        outputs = outputs.squeeze(1)
        outputs = self.cls_head(outputs)
        return outputs
        
        
if __name__ == "__main__":
    model = CNN_Transformer()
    print(model)
    inputs = torch.randn(8, 39, 256)
    
    output = model(inputs, None)
    print(output.shape)