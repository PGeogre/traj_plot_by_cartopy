import torch
import torch.nn as nn
import math
from readdata import MyDataset
from torch.utils.data import DataLoader
import numpy as np

# 进行位置编码
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.pe[:, :x.size(1)]

class LatLonEmbedding(nn.Module):
    def __init__(self, d_model):
        super(LatLonEmbedding, self).__init__()
        self.conv_layer_3tod = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
    def geodetic_to_ecef(self, lat_rad, lon_rad):
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return x,y,z

    def forward(self, x):
        lat = x[:,:,1]
        lon = x[:,:,2]
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        x, y, z = self.geodetic_to_ecef(lat_rad, lon_rad)  # (N,512)
        xyz = torch.stack([x, y, z], dim=1)  # [B,3,512]
        features = self.conv_layer_3tod(xyz.float()) # [B,128,512]
        features = features.transpose(1, 2)
        return features

class SOGCOGEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SOGCOGEmbedding, self).__init__()
        self.cos = torch.cos
        self.sin = torch.sin
        self.conv_layer_2tod = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        SOG = x[:,:,3]
        COG = x[:,:,4]
        COG_rad = torch.deg2rad(COG)
        SOG_1 = self.sin(COG_rad) * SOG
        SOG_2 = self.cos(COG_rad) * SOG

        xyz = torch.stack([SOG_1, SOG_2], dim=1)  # [B,2,L]
        features = self.conv_layer_2tod(xyz.float()) # [B,128,L]
        features = features.transpose(1, 2) # [B,128,L]
        return features

class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__()
        self.dv = d_model
        self.half_dv = d_model // 2
        self.omega = nn.Parameter(torch.randn(self.half_dv))

    def forward(self, x):

        self.omega = self.omega.to(x.device)
        t = x[:,:,0]
        zt = self.continuous_encoding(t)
        # print("zt shape:",zt.shape)
        return zt
    def continuous_encoding(self, v):

        v = v.unsqueeze(-1)
        encoding_cos = torch.cos(self.omega * v)
        encoding_sin = torch.sin(self.omega * v)

        encoding = torch.zeros(v.size(0), v.size(1), self.dv)
        encoding[:, :, 0::2] = encoding_cos
        encoding[:, :, 1::2] = encoding_sin
        return encoding


class TrajEmbedding(nn.Module):
    """DataEmbedding embedding"""

    def __init__(self, d_model, dropout=0.1):
        super(TrajEmbedding, self).__init__()

        self.d_model = d_model
        self.LatLonEmbedding = LatLonEmbedding(d_model=self.d_model)
        self.SOGCOGEmbedding = SOGCOGEmbedding(d_model=self.d_model)
        self.TimeEmbedding = TimeEmbedding(d_model=self.d_model)
        self.position_embedding = PositionalEmbedding(d_model=self.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        device = x.device

        x = self.position_embedding(x).to(device) + self.TimeEmbedding(x).to(device)+\
            self.LatLonEmbedding(x).to(device) + self.SOGCOGEmbedding(x).to(device)

        return self.dropout(x)

if __name__ == '__main__':
    Dataset = MyDataset('NJ_data/vaild')
    print(len(Dataset))
    dataloader = DataLoader(Dataset, batch_size=4, shuffle=True)
    for i,(a,b) in enumerate(dataloader):
        print("input shape:",a.shape)
        timeembedding = TimeEmbedding(d_model=128)
        time_out = timeembedding(a)
        print("time out",time_out.shape)
        latlon = LatLonEmbedding(d_model=128)
        latlon_out = latlon(a)
        print("lat out",latlon_out.shape)

        sogembedding = SOGCOGEmbedding(d_model=128)
        sog_out = sogembedding(a)
        print("sog out",sog_out.shape)

        pos = PositionalEmbedding(d_model=128)
        pos_out = pos(a)
        print("pos out :",pos_out.shape)

        embed = TrajEmbedding(d_model=128)
        embed_out = embed(a)
        print("embed_out out :", embed_out.shape)


