import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from embedding import TrajEmbedding
from torch.utils.data import Dataset, DataLoader


class TrajGPT_Base(nn.Module):
    """TrajGPT"""

    def __init__(self, enc_in=5,d_model=128):
        super(TrajGPT_Base, self).__init__()
        self.enc_embedding = TrajEmbedding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128,nhead=2,dropout=0.1,dim_feedforward=256,
                                                        activation="gelu",batch_first=True,dtype=torch.float32)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=2)
        self.att=OneEncoder(d_model)

        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, num_classes)
        # )

    def forward(self, input):

        input = input.float() # (B,L,c_in)
        batch_size, seq_len, d_model = input.size()
        emb_out = self.enc_embedding(input) # (B,L,D)

        # 通过encoder提取之后得到的特征__!!(用于下游任务)
        base_out = self.transformer_encoder(emb_out) # (B,L,D)

        # (B,L,D) to (B,1,D)
        att_out = self.att(base_out) # (B,1,D)
        att_out = att_out.squeeze(1) # (B,,D)


        """
        分类
        enc_out = emb_out.transpose(1,2) # (B,D,L)
        pool_out = self.pool(enc_out) # (B,D,1)
        pool_out = pool_out.squeeze() # (B,D)
        out = self.classifier(pool_out) # (B,num_class)
        """

        return base_out,att_out



class OneEncoder(nn.Module):
    def __init__(self, input_dim):
        super(OneEncoder, self).__init__()
        self.d_model = input_dim

        self.encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1)

        self.add_feature = nn.Parameter(torch.randn(1, 1, self.d_model))

    def forward(self, x):
        batch_size, seq_len, d_model = x.size() # (B,L,D)
        # (1,1,D) to (B,1,D)

        add_feature = self.add_feature.expand(x.shape[0],-1,-1)
        add_feature = add_feature.to(x.device)
        x = torch.cat((x, add_feature), dim=1)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, dim)
        x = self.encoder(x) # (L+1,B,D)
        x = x.permute(1, 0, 2)
        x = x[:,-1,:]

        return x


