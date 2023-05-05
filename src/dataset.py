from torch.utils.data import Dataset
from src.utils import spec_augment

class SER_Dataset(Dataset):
    def __init__(self, inputs, labels, mode="train"):
        self.inputs = inputs
        self.labels = labels
        self.mode = mode
    def __len__(self):
        return self.inputs.shape[0]
    
    def smooth_labels(self, labels, factor=0.1):
        # smooth the labels
        labels *= (1 - factor)
        labels += (factor / labels.shape[0])
        return labels
    
    def __getitem__(self, index):
        if self.mode == "train":
            inputs = self.inputs[index]
            inputs = spec_augment(inputs)
        else:
            inputs = self.inputs[index]
        labels = self.labels[index]
        labels = self.smooth_labels(labels)
        
        return inputs, labels