from torch.nn.utils import weight_norm
import torch.nn.functional as F 
from torch import nn
import torch
from .conformer.encoder import ConformerBlock
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        channels = [39, 128]
        kernels = [3, 3]
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
    
class ConformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 128,
            num_layers: int = 4,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual) 
                for _ in range(num_layers)]
            )
        
    def forward(self, inputs, lengths=None):
        hiddens = []
        for layer in self.layers:
            inputs = layer(inputs)
            
            tmp = inputs.mean(dim=1).unsqueeze(1)
            hiddens.append(tmp)
        hiddens = torch.cat(hiddens, dim=1)
        return inputs, hiddens
    

class CNN_Conformer(nn.Module):
    def __init__(self, config=None, num_layers=4) -> None:
        super(CNN_Conformer, self).__init__()
        self.cnn = CNN()
        self.weighted_layers = nn.Parameter(torch.randn(1, 4))
        self.conformer = ConformerEncoder(
            encoder_dim=128,
            num_layers=num_layers,
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.2,
            attention_dropout_p=0.2,
            conv_dropout_p=0.2,
            conv_kernel_size=15,
            half_step_residual=True
        )
        self.cls_head = nn.Linear(128, 8)
        self.weighted_layers = nn.Parameter(torch.randn(1, num_layers))

    def forward(self, inputs, lengths):
        inputs = self.cnn(inputs)
        inputs = inputs.transpose(1, 2)
        inputs, hiddens = self.conformer(inputs)
        
        outputs = torch.matmul(self.weighted_layers, hiddens)
        outputs = outputs.squeeze(1)
        outputs = self.cls_head(outputs)

        return outputs
        
        
if __name__ == "__main__":
    model = CNN_Conformer()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(model)
    print("num params: ",params)
    inputs = torch.randn(8, 39, 256)
    
    output = model(inputs, None)
    print(output.shape)