import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from readdata import MyDataset
from finetunemodel import FineTunedTrajGPT

# dataset
train_dataset = MyDataset("NJ_data/train")
val_dataset = MyDataset("NJ_data/vaild")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True,drop_last=True)


# 初始化模型
model = FineTunedTrajGPT('model_path/base/base_model.pth', d_model=128, num_classes=14)
model.to('cuda:0')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# optimizer = torch.optim.AdamW(model.classifier2.parameters(), lr=1e-3)

def save_model(model, epoch, saved_models, max_saved_models=5, model_dir='model_path'):
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(model_dir, f'class_model_{epoch}.pth')
    # torch.save(model.state_dict(), model_save_path)
    torch.save(model, model_save_path)
    print(f"Saved model at {model_save_path}")

    saved_models.append(model_save_path)

    if len(saved_models) > max_saved_models:
        oldest_model = saved_models.pop(0)
        os.remove(oldest_model)
        print(f"Removed old model at {oldest_model}")

def train():
    saved_models = []
    num_epochs = 500
    best_val_acc = 0.0

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for idx,(inputs, labels) in enumerate(train_dataloader):
            start_time = time.time()
            # inputs = inputs.float()
            # labels = labels.float()
            inputs = inputs.to('cuda:0')
            labels = labels.to('cuda:0')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(f'epoch{epoch}-step{idx}-loss{loss.item()}')
            loss.backward()
            optimizer.step()
            end_time = time.time()

            with open(os.path.join('logs', "log.txt"), 'a+') as file:
                file.write(
                    f"epoch{epoch}-step{idx}-loss{loss:.4e}-usetime{start_time - end_time}s\n")

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct/train_total
        train_loss = train_loss/len(train_dataloader)
        print(f'epoch{epoch}-train_acc{train_acc}-train_loss{train_loss}\n')
        with open(os.path.join('logs', "train_acc.txt"), 'a+') as file:
            file.write(
                f'epoch{epoch}-train_acc{train_acc}-train_loss{train_loss}\n')

        save_model(model, epoch, saved_models)

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                # val_inputs = val_inputs.float()
                val_inputs = val_inputs.to('cuda:0')
                val_labels = val_labels.to('cuda:0')

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_dataloader)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n,'
            f' Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        with open(os.path.join('logs', "val_acc.txt"), 'a+') as file:
            file.write(
                f'epoch{epoch}-val_Loss: {val_loss:.4f}, val_Acc: {val_acc:.2f}%\n')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_path/best_class_model.pth')
            torch.save(model, 'model_path/full_best_class_model.pth')

if __name__ == '__main__':
    train()