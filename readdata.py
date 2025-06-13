import os
import csv
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, folder_path):
        super(MyDataset, self).__init__()
        self.input = []
        self.labels = []
        self.label_to_idx = {}
        self.load_data(folder_path)

    def load_data(self, folder_path):
        subfolders = glob.glob(os.path.join(folder_path, '*'))
        for i, subfolder in enumerate(subfolders):
            self.label_to_idx[os.path.basename(subfolder)] = i
            csv_files = glob.glob(os.path.join(subfolder, '*.csv'))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S').astype("int64")//10**9
                # 提取时间特征

                df_feature = df[['date','lat', 'lon', 'sog', 'cog']]

                num_rows = df.shape[0]
                num_chunks = num_rows // 512
                # 取整1024块
                for i in range(num_chunks):
                    start = i * 512
                    end = start + 512
                    chunk1 = df_feature[start:end].values.tolist()
                    self.input.append(chunk1)
                    self.labels.append(self.label_to_idx[os.path.basename(subfolder)])

                remain_rows = num_rows % 512
                if 100 < remain_rows < 512:
                    start = num_chunks * 512
                    end = start + remain_rows
                    chunk = df_feature[start:end].values.tolist()

                    padding_rows = 512 - remain_rows
                    padding_data = [[0] * len(df_feature.columns)] * padding_rows
                    chunk += padding_data
                    self.input.append(chunk)
                    self.labels.append(self.label_to_idx[os.path.basename(subfolder)])

        self.input = np.array(self.input)
        self.input = torch.from_numpy(self.input)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.labels[idx]

if __name__ == "__main__":
    dataset = MyDataset("NJ_data/vaild")
    print(len(dataset)) # 3191 # 3458
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    for inputs, labels in dataloader:
        print("Inputs shape:", inputs.shape)
        print("Labels shape:", labels)

