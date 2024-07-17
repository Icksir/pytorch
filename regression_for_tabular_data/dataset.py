from torch.utils.data import Dataset
import torch
import numpy as np

class ConsumoDataset(Dataset):
    def __init__(self, df):
        self.y = df.pop('consumo_max')
        self.x = df

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.x.iloc[index].astype(np.float32)
        y = self.y.iloc[index]

        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor