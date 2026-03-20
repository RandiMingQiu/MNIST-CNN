from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

#数据集的处理
transform=transforms.ToTensor()
dataset=datasets.MNIST(
    root='./data',
    transform=transform,
    train=True,
    download=True,
)
train_dataloader=DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

#神经网络的定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,100)
        self.relu=nn.ReLU()#relu本质上是一个取最大值的函数
        self.fc2=nn.Linear(100,10)

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

#训练参数的设定
learningrate=0.01
epochs=15
total_loss=0
loss_list=[]

#参数更新对象函数与loss计算函数的定义
model=Net()
model.load_state_dict(torch.load('Mnist_model.pth'))
optimizer=optim.SGD(model.parameters(),lr=learningrate)
criterion=nn.CrossEntropyLoss()#交叉熵损失函数

#训练循环主体，加入了loss平均值的计算
for epoch in range(epochs):
    for x,y in train_dataloader:
        x=x.view(x.size(0),-1)
        optimizer.zero_grad()
        output=model(x)
        loss=criterion(output,y)
        loss.backward()
        optimizer.step()
        total_loss=total_loss+loss.item()
    avg_loss=total_loss/len(train_dataloader)
    print(epoch+1,":",avg_loss)
    loss_list.append(avg_loss)
    total_loss=0

#画loss图
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

#保存模型的参数，这句话会在当前目录生成名为Mnist_model.pth的二进制文件
torch.save(model.state_dict(), "Mnist_model.pth")
print("此次训练参数已保存")
