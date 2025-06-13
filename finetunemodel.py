import torch
import torch.nn as nn
from readdata import MyDataset
from torch.utils.data import Dataset, DataLoader


class FineTunedTrajGPT(nn.Module):
    def __init__(self, pretrained_model_path, d_model, num_classes):
        super(FineTunedTrajGPT, self).__init__()

        self.traj_gpt = torch.load(pretrained_model_path,map_location='cpu')

        for param in self.traj_gpt.parameters():
            param.requires_grad = False

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layers,num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, input_data):

        # encoder (B,L,D),(B,D)
        logits,features = self.traj_gpt(input_data)
        logits = logits.permute(1,0,2) # (L,B,D)
        output = self.encoder(logits) # (L,B,D)
        # pool
        output = output.permute(1,2,0) # (B,D,L)

        pool_out = self.pool(output) # (B,D,1)

        pool_out = pool_out.squeeze(-1)  # (B,D)
        out = self.classifier(pool_out) # (B,num_class)

        return out


if __name__=='__main__':

    model = FineTunedTrajGPT('model_path/base/base_model.pth',d_model=128, num_classes=14)
    model = model.to('cuda:0')

    dataset = MyDataset("NJ_data/vaild")
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    for inputs, labels in dataloader:
        inputs = inputs.float()
        inputs = inputs.to('cuda:0')
        print("Inputs shape:", inputs.shape) # (16,512,5)
        print("Labels shape:", labels.shape)# (16)
        print("labels",labels)
        out = model(inputs) # (16,3)
        print("out shape:",out.shape)
        print(out)
        # _, predicted = torch.max(out.data, 1)
        # print("pred:",predicted)


