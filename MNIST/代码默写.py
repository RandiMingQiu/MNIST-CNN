from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
#数据集的预处理
transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])

dataset=datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)

dataloader=DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

#CNN的定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,20,3)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2)
        self.fc1=nn.Linear(20*5*5,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.dropout=nn.Dropout(0.4)

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.fc3(x)
        x=self.relu(x)

        return x

#定义规则与参数
filename="MNIST_Pro_digits.pth"
epochs=20
learningrate=0.01
total_loss=0
model=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learningrate)

#开始训练
for epoch in range(epochs):
    for x,y in dataloader:
        optimizer.zero_grad()
        output=model(x)
        loss=criterion(output,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(dataloader)
    total_loss=0
    print("epoch:",epoch+1,"loss:",avg_loss)

torch.save(model.state_dict(),filename)
print("本次训练完成，模型参数已保存至：",filename)














