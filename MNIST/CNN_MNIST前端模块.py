import cv2
import torch.nn as nn
import torch
import torchvision.transforms as transforms


#定义神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,3)
        self.conv2=nn.Conv2d(10,20,3)
        self.relu=nn.ReLU()
        self.pooling=nn.MaxPool2d(2)
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

        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc3(x)

        return x

#定义模型与加载参数进入模型,把模型调整到预测模式,参数控制
model=Net()
model.load_state_dict(torch.load("CNN_MNIST_modelpro.pth"))#这是微调过的参数文件
model.eval()
recent_preds=[]
N=5

#定义图像预处理
transform=transforms.Compose([transforms.ToTensor()])

#打开摄像头,o是默认摄像头
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    print("摄像头打开失败，请检查驱动！")
    exit()


#实时监测的循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 自适应二值化（比固定阈值更鲁棒，适配不同光线）
    img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2  # 块大小11，常数2，可根据实际调整
    )
    # Resize到28×28（MNIST输入尺寸）
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # 归一化到0-1（和ToTensor的效果对齐）
    img = img / 255.0

    # 3. 转换为模型输入格式（添加batch维度+标准化）
    x = transform(img).unsqueeze(0).float()  # (1,1,28,28)，符合模型输入维度

    # 模型预测
    with torch.no_grad():  # 不计算梯度，节省资源
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    #把最近N帧的结果存入列表，并删除最旧的元素，注意：列表在 Python 里是动态数组，删除元素后，剩下的元素会自动向前挪位，索引自动更新。
    recent_preds.append(pred)
    if len(recent_preds)>N:
        recent_preds.pop(0)

    #投票得出过去几帧出现最多的数字，实现无空窗期的丝滑预测
    final_pred=max(set(recent_preds),key=recent_preds.count)

    # 显示预测结果
    cv2.putText(
        frame,
        f'Figure:{final_pred}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.imshow('Handwriting Recognition', frame)

    # 按英文 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#释放缓存资源
cap.release()
cv2.destroyAllWindows()

