import torch.nn as nn

class HornoModel(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, dropout=0.25):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden_1)
        self.linear2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.linear3 = nn.Linear(n_hidden_2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        return x