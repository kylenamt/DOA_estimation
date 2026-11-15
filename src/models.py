import torch.nn as nn
import torch.nn.functional as F


class BaseDOAModel(nn.Module):
    def __init__(self, d_model, ):
        super(BaseDOAModel, self).__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.BatchNorm1d(64),  
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.classification_layer = nn.Sequential(
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )
        self.DOA_1 = nn.Sequential(
            nn.Linear(256, 1),)
        self.DOA_2 = nn.Sequential(
            nn.Linear(256, 2),)
        self.DOA_3 = nn.Sequential(
            nn.Linear(256, 3),)
        self.DOA_4 = nn.Sequential(
            nn.Linear(256, 4),)

    def forward(self, x):
        x = self.net(x)
        classification_output = self.classification_layer(x)
        doa_1_output = self.DOA_1(x)
        doa_2_output = self.DOA_2(x)
        doa_3_output = self.DOA_3(x)
        doa_4_output = self.DOA_4(x)

        return classification_output, doa_1_output, doa_2_output, doa_3_output, doa_4_output
        
            
            