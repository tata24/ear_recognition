import torch
from torch import nn
import os
from dataset import MyDataset
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

"""超参数设置"""
USE_GPU = True
batch_size = 50
dtype = torch.float32
learning_rate = 1e-3
epoch = 100
device = torch.device('cpu')
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')


"""构造dataset和dataloader"""
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
root = os.path.abspath(os.path.dirname(__file__)) + '/data'
myDataset = MyDataset(root=root, transforms=data_transforms)
print('数据集总容量：', len(myDataset))
train_data, val_data = torch.utils.data.random_split(myDataset, [1000, 200])
print('划分数据集：', '训练集容量：', len(train_data), '验证集容量：', len(val_data))
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)
# 损失函数
criterion = nn.CrossEntropyLoss()

train_accuracy = np.zeros(epoch)
val_accuracy = np.zeros(epoch)


def plot_train_val_curve():
    plt.figure()
    plt.plot(list(range(1, epoch + 1)), train_accuracy, label='train')
    plt.plot(list(range(1, epoch + 1)), val_accuracy, label='validation')
    plt.show()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = torch.max(scores, 1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc


def train(model, optimizer, epoch):
    for i in range(epoch):
        model = model.to(device=device)
        img = 0
        for t, (x, y) in enumerate(train_loader):
            model.train(mode=True)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            img = img + len(y)
            optimizer.zero_grad()
            scores = model(x)
            loss = criterion(scores, y)
            loss.backward()
            optimizer.step()
            print('epoch %d: Iteration %d, %d / %d, loss = %.4f, validation accuracy: %.2f' %
                  (i+1, t, img, len(train_data), loss.item(), check_accuracy(val_loader, model)))
        train_accuracy[i] = check_accuracy(train_loader, model)
        val_accuracy[i] = check_accuracy(val_loader, model)


if __name__ == '__main__':
    # 构建模型
    resnet = models.resnet50(pretrained=False)
    # 多GPU训练
    resnet = nn.DataParallel(resnet)
    optimizer = optim.SGD(resnet.parameters(), lr=learning_rate,
                          momentum=0.9, nesterov=True)
    # 开始训练
    train(model=resnet, optimizer=optimizer, epoch=epoch)
    plot_train_val_curve()
    torch.save(resnet.state_dict(), './result/result.pth')





