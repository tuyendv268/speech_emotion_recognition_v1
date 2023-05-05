import torch.nn.functional as F
from yaml.loader import SafeLoader
from torch.nn.utils import weight_norm
from torch import nn
import numpy as np
import torch
import yaml

class AdditiveAttention(nn.Module):
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, dropout=None):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        if dropout is not None:
            candidate_weights = dropout(candidate_weights)
            
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, Q, K, V, attention_masks=None, dropout=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attention_masks is not None:
            scores = scores * attention_masks
        attention_scores = scores/(torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        context = torch.matmul(attention_scores, V)
        return context, attention_scores

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, attention_dim, num_attention_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        
        self.d_k = attention_dim
        self.d_v = attention_dim
        
        self.W_Q = nn.Linear(d_model, attention_dim * num_attention_heads)
        self.W_K = nn.Linear(d_model, attention_dim * num_attention_heads)
        self.W_V = nn.Linear(d_model, attention_dim * num_attention_heads)
        
        self.ffw = nn.Linear(attention_dim * num_attention_heads, d_model)
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
        
        context, attention_weights = ScaledDotProductAttention(self.d_k)(
            Q=q_s, K=k_s, V=v_s,
            attention_masks=attention_masks,
            dropout=self.dropout
        )
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, 
            self.num_attention_heads * self.d_v)
        context = self.ffw(context)
        return context

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        self.config = config
        channels = config["conv"]["channels"]
        self.left_paddings = config["conv"]["left_paddings"]
        convs = []
        
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            if i < config["num_pooling"]:
                conv = [
                    weight_norm(nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=config["conv"]["kernels"][i],
                        padding=config["padding"],
                        stride=config["conv"]["strides"][i])),
                    nn.BatchNorm1d(num_features=out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(
                        kernel_size=config["pooling"]["kernels"][i], 
                        # padding=config["padding"],
                        stride=config["pooling"]["strides"][i])
                ]
            else:
                conv = [
                    weight_norm(nn.Conv1d(
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        kernel_size=config["conv"]["kernels"][i],
                        padding=config["padding"],
                        stride=config["conv"]["strides"][i])),
                    nn.BatchNorm1d(num_features=out_channels),
                    nn.ReLU(),
                ]
            
            convs += conv
        
        convs = nn.Sequential(*convs)
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(config["drop_rate"])
            
    def _conv_out_length(self, input_length, kernel_size, stride):
        return torch.div(input_length, stride, rounding_mode="floor") + 1
    
    def _get_feat_extract_output_lengths(self, input_lengths):
        conv_kernels, conv_strides = self.config["conv"]["kernels"], self.config["conv"]["strides"]
        for i, (kernel_size, stride) in enumerate(zip(conv_kernels, conv_strides)):
            input_lengths = self._conv_out_length(
                input_lengths, kernel_size, stride)
            
            if i < self.config["num_pooling"]:
                input_lengths = self._conv_out_length(
                    input_lengths, 
                    self.config["pooling"]["kernels"][i], 
                    self.config["pooling"]["strides"][i])
        return input_lengths
    
    def forward(self, inputs):
        count = 0
        for conv in self.convs:
            if type(conv).__name__ == "Conv1d":
                inputs = F.pad(inputs.unsqueeze(2), (self.left_paddings[count], 0, 0, 0)).squeeze(2)
                count += 1
            inputs = conv(inputs)
        
        inputs = self.dropout(inputs)
        return inputs

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        
        self.config = config
        self.rnn2module = {
            "gru":nn.GRU,
            "lstm":nn.LSTM
        }
        rnn = self.rnn2module[config["rnn"]]
        
        if config["bidirectional"] == True:
            self.rnn = rnn(
                input_size = config["input_dim"],
                hidden_size=config["hidden_dim"]//2,
                batch_first=True,
                dropout=config["drop_rate"],
                bidirectional=True
            )
        else:
            self.rnn = rnn(
                input_size = config["input_dim"],
                hidden_size= config["hidden_dim"],
                batch_first=True,
                dropout=config["drop_rate"],
                bidirectional=False
            )

    def forward(self, inputs):
        inputs, _ = self.rnn(inputs)
        return inputs
  
class CNN_RNN_SER(nn.Module):
    def __init__(self, config):
        super(CNN_RNN_SER, self).__init__()
        
        self.config = config
        self.device = "cpu" if not torch.cuda.is_available() else config["device"]
        
        self.convs = CNN(
            config=config["cnn"]
            )    
        
        self.rnn = RNN(
            config=config['rnn']
            )
        
        self.multi_head_self_attentions = MultiHeadSelfAttention(
            d_model=config["attention"]["self"]["hidden_dim"],
            attention_dim=config["attention"]["self"]["hidden_dim"],
            num_attention_heads=config["attention"]["self"]["num_head"],
            dropout=config["attention"]["self"]["drop_rate"]
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim=config["attention"]["additive"]["query_dim"],
            candidate_vector_dim=config["attention"]["additive"]["hidden_dim"]
        )
        
        self.dropout = nn.Dropout(config["drop_rate"])
        self.linear = nn.Linear(config["hidden_dim"], config["num_labels"])

    
    def forward(self, **kwargs):
        inputs, lengths = kwargs["features"], kwargs["lengths"]
        inputs = self.convs(inputs)
        inputs = inputs.transpose(-1, -2)
        inputs = self.rnn(inputs)
        
        lengths = self.convs._get_feat_extract_output_lengths(lengths)
        max_length = inputs.size(1)

        attention_masks = torch.arange(max_length).expand(inputs.size(0), inputs.size(1)).to(self.device)
        attention_masks = attention_masks < lengths.unsqueeze(1)
        # attention_masks = torch.ones_like(inputs).to(self.device)
        
        inputs = self.multi_head_self_attentions(inputs, attention_masks)
        inputs = self.dropout(inputs)
        
        outputs = self.additive_attention(inputs, self.dropout)
        outputs = self.linear(outputs)
        return outputs
    
if __name__ == "__main__":
    with open('../../configs/models/cnn_rnn_attn.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)
    model = CNN_RNN_SER(config=config)
    print(model)

    inputs = torch.randn(8, 80, 86)
    lengths = torch.randint(12, 86, (8,))
    outputs = model(features=inputs, lengths=lengths)
    print(outputs.shape)