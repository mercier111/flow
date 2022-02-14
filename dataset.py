import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset, DataLoader


class trainset(Dataset):
    def __init__(self, csv_path=[]):
        self.csv_path = csv_path
        self.len = []
        self.data = []
        for path in csv_path:
            data = pd.read_csv(path)
            self.len.append(len(data))
            self.data.append(data)

    def __getitem__(self, index):
        print(index)
        for i, len in enumerate(self.len):
            if index < len :
                data = self.data[i]
                break
            index -= len

        input  = np.array(data.iloc[index, :-1])
        output = np.array(data.iloc[index, -1])
        return input, output

    def __len__(self):
        return sum(self.len)