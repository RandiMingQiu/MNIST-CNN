from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

#图片的预处理
transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#数据的集合，
dataset=datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True,
)
#分批送入，64张为一批
train_loader=DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

#CNN定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,20,3)
        self.pooling=nn.MaxPool2d(2)
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(20*5*5,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.dropout=nn.Dropout(0.4)#每次训练随机关闭40%的神经元，防止过拟合

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pooling(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pooling(x)

        x=x.view(x.size(0),-1)#拉平送入全连接层
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.fc3(x)

        return x

#参数的设定
learningrate=0.01
epochs=15
total_loss=0
loss_list=[]
#定义参数更新器与loss计算器,加载最新参数
model=Net()
model.load_state_dict(torch.load('CNN_MNIST_model.pth'))
optimizer=optim.SGD(model.parameters(),lr=learningrate)
criterion=nn.CrossEntropyLoss()#交叉熵损失函数计算loss

#训练循环
for epoch in range(epochs):
    model.train()#养成好习惯
    for x,y in train_loader:
        optimizer.zero_grad()
        output=model(x)
        loss=criterion(output,y)
        loss.backward()
        optimizer.step()
        total_loss=total_loss+loss.item()
    avg_loss=total_loss/len(train_loader)
    print(epoch+1,avg_loss)
    loss_list.append(avg_loss)
    total_loss=0

#画loss图
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

#保存模型的参数到同一根目录下的二进制文件
torch.save(model.state_dict(), "CNN_MNIST_model.pth")
print("模型参数已保存")


