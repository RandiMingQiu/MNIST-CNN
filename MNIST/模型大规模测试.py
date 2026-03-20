from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch


transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset=datasets.MNIST(
    root='./data',
    download=True,
    transform=transform,
    train=False,#测试要用没见过的数据
)
test_loader=DataLoader(
    dataset,
    batch_size=100,
    shuffle=True,
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,20,3)
        self.pool=nn.MaxPool2d(2)
        self.fc1=nn.Linear(20*5*5,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.4)

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)

        x=self.fc3(x)

        return x

model=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
count=0

with torch.no_grad():
    for i in range(2):
        filename=input("请输入模型参数文件名（含后缀）：")
        model.load_state_dict(torch.load(filename))
        model.eval()
        count=0
        total=0
        for x,y in test_loader:
            output=model(x)
            pred=torch.argmax(output,dim=1)
            # .item()的作用就是把张量里面的数值变成python可以直接用的普通数字
            #pre=y，列表里面有100个布尔值，sum计数多少个true,交给item()转化
            count=(pred == y).sum().item()+count
            total=total+y.size(0)
        print("模型正确率：",float(count/total)*100,'%',"参数来源：",filename)




