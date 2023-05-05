from yaml.loader import SafeLoader
from torch.nn.utils import weight_norm
from torch import nn
import yaml
import torch

class Feature_Extraction_CNN(nn.Module):
    def __init__(self):
        super(Feature_Extraction_CNN, self).__init__()

        channels = [32*3, 64, 96, 128, 160, 320]
        kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (1, 1)]
        strides = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        avg_pooling_kernels = [(2, 2), (2, 2), (2, 1), (2, 1), None]
        avg_pooling_strides = [(2, 2), (2, 2), (2, 1), (2, 1), None]
        convs = []
        
        for i, (in_channel, out_channel) in enumerate(zip(channels[:-1], channels[1:])):
            if avg_pooling_strides[i] is not None:
                conv = [
                    weight_norm(nn.Conv2d(
                        in_channels=in_channel, 
                        out_channels=out_channel, 
                        kernel_size=kernels[i],
                        padding="same",
                        stride=strides[i])),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.AvgPool2d(
                        kernel_size=avg_pooling_kernels[i], 
                        stride=avg_pooling_strides[i],
                        padding=0
                        ),
                ]
            else:
                conv = [
                    weight_norm(nn.Conv2d(
                        in_channels=in_channel, 
                        out_channels=out_channel, 
                        kernel_size=kernels[i],
                        padding="same",
                        stride=strides[i])),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                ]
            convs += conv
        
        convs = nn.Sequential(*convs)
        self.convs = nn.ModuleList(convs)
        self.drop_out = nn.Dropout(0.3)
    
    def forward(self, inputs):
        for conv in self.convs:
            inputs = conv(inputs)
        
        inputs = torch.mean(inputs, dim=1)
        inputs = self.drop_out(inputs)
        inputs = torch.flatten(inputs, start_dim=1)
        return inputs
    
class Parallel_CNN(nn.Module):
    def __init__(self):
        super(Parallel_CNN, self).__init__()
        self.temporal_cnn = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=1,
                    padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ]
        )
        self.spectral_cnn = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(9, 1),
                    stride=1,
                    padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ]
        )
        
        self.origin_cnn = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(1, 11),
                    stride=1,
                    padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ]
        )
        
        self.avg_pooling = nn.AvgPool2d(
            kernel_size=(2, 2),
            padding="same",
            stride=2
        )
        
    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        temporal_out = self.temporal_cnn(inputs)
        origin_out = self.origin_cnn(inputs)
        spectral_out = self.spectral_cnn(inputs)
        
        output = torch.cat([temporal_out, origin_out, spectral_out], dim=1)
        return output

class Light_SER(nn.Module):
    def __init__(self, config=None) -> None:
        super(Light_SER, self).__init__()
        
        self.parallel_cnn = Parallel_CNN()
        self.feature_extractor_cnn = Feature_Extraction_CNN()
        self.ffw = nn.Linear(106, 8)
        
    def forward(self, inputs, lengths=None):
        inputs = self.parallel_cnn(inputs)
        inputs = self.feature_extractor_cnn(inputs)
        inputs = self.ffw(inputs)
        
        return inputs
    
if __name__ == "__main__":
    model = Light_SER()
    print(model)
    inputs = torch.randn(8, 39, 256)    
    output = model(features=inputs)
    print(output.shape)