#使用模型进行预测
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 1. 模型结构（必须一致）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x=self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2. 创建模型并加载参数
model = Net()
model.load_state_dict(torch.load("Mnist_model.pth"))
model.eval()


# 3. 准备一张测试图片（从数据库中随机抽取）
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


x, y = test_dataset[12]   # 一张图
x = x.view(x.size(0), -1)        # [1, 784]


# 4. 推理
with torch.no_grad():
    output = model(x)
    pred = output.argmax(dim=1)

#输出预测结果
print("预测结果:", pred.item())
print("真实标签:", y)

