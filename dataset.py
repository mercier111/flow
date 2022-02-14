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
        for i, len in enumerate(self.len):
            if index < len :
                data = self.data[i]
                break
            index -= len

        input = np.array(data.iloc[index, :-1])
        input = self.normalize(input)
        output = np.array(data.iloc[index, -1])
        return input, output

    def __len__(self):
        return sum(self.len)

    def normalize(self, input):
        i1, i2, i3, i4, i5, i6, i7 = input
        i1 *= 100
        i2 *= 200
        i3 /= 1000
        i4 *= 100
        i5 /= 5
        i6 *= 10000000
        i7 *= 10000000
        input = np.array([i1, i2, i3, i4, i5, i6, i7])

        return input