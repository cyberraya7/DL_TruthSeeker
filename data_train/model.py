import torch.nn as nn
import torch

class IDSModel(nn.Module):
    def __init__(self, input_features, num_classes):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)
    

    def forward(self, x):
        
        
        x = self.fc1(x)
        x = self.batch_norm1(x)  
        x = self.activation1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)  
        x = self.activation2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x