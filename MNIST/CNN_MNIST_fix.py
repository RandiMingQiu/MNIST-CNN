import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 图像预处理：灰度化 + 转 tensor
transform = transforms.Compose([
    transforms.Grayscale(),  # 确保灰度
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载新采集的数据
dataset = ImageFolder(
    root="D:\\PythonAI\\MNIST\\MNIST_Pro_digits",
    transform=transform
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=12,
    shuffle=True
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
        self.dropout=nn.Dropout(0.4)

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


# 设置为训练模式,加载原始 MNIST 模型
model = Net()
model.load_state_dict(torch.load("CNN_MNIST_modelpro.pth"))
model.train()

# 损失函数和优化器, 学习率要小，避免破坏原模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
epochs = 10 # 微调不需要多轮

for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        # x: [batch, 1, 28,28]，y: label
        # 如果拍摄的图片是白底黑字，不用 worry，后面已经 threshold 了

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, loss={total_loss / len(dataloader):.4f}")

# 保存微调后的模型
torch.save(model.state_dict(), "CNN_MNIST_modelpro.pth")
print("微调完成！模型已保存！")