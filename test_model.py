import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, csv_file):
        super(MyDataset, self).__init__()
        self.input = []
        self.load_data(csv_file)

    def load_data(self, csv_file):
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S').astype("int64") // 10 ** 9  # 转换为时间戳

        df_feature = df[['date', 'lat', 'lon', 'sog', 'cog']].head(512)

        self.input = df_feature.values.tolist()

        if len(self.input) < 512:
            padding_rows = 512 - len(self.input)
            padding_data = [[0] * len(df_feature.columns)] * padding_rows
            self.input += padding_data

        self.input = np.array(self.input)
        self.input = torch.from_numpy(self.input)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.input


def save_predictions_to_csv(csv_file, prediction):
    df = pd.read_csv(csv_file)

    df['label'] = prediction
    df.to_csv(csv_file, index=False)


import os
import pandas as pd

import os
import pandas as pd


def map_labels_to_chinese(folder_path):
    # 定义映射表
    label_mapping = {
        0: '散货船',
        1: '货船',
        2: '集装箱船',
        3: '驳船',
        4: '渔船',
        5: '其他',
        6: '油船',
        7: '客船',
        8: '运沙船',
        9: '鱼类实验船',
        10: '补给舰',
        11: '储罐船',
        12: '潜艇',
        13: '运输船'
    }

    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 映射 label 列
            if 'label' in df.columns:
                df['label'] = df['label'].map(label_mapping)

            # 保存修改到原 CSV 文件
            df.to_csv(file_path, index=False, encoding='utf-8')
            # print(f'Processed and updated {filename}')


def test(folder_path):
    # folder_path = "NJ_data/test_data"
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    model = torch.load("model_path/full_best_class_model.pth",map_location='cpu')

    model.eval()

    for csv_file in csv_files:
        full_path = os.path.join(folder_path, csv_file)
        dataset = MyDataset(full_path)
        original_df = pd.read_csv(full_path)
        original_length = len(original_df)
        dataloader = DataLoader(dataset, batch_size=1)  #

        predictions = []

        with torch.no_grad():
            for inputs in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item())

        label_prediction = predictions[0]
        label_list = [label_prediction] * original_length

        save_predictions_to_csv(full_path, label_list)

    # make label
    map_labels_to_chinese(folder_path)


    # return "Class Model test finished!"

if __name__ == '__main__':
    path = "NJ_data/test_data"
    test(path)