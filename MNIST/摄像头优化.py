import cv2
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import numpy as np

# 必须和训练时的模型结构完全一致，否则加载参数报错
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20*5*5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.4)  # 必须和训练时完全一样

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 和训练时一致

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # 和训练时一致

        x = self.fc3(x)
        return x

# ===================== 2. 加载模型（修正结构+预测模式） =====================
model = Net()
try:
    model.load_state_dict(torch.load("CNN_MNIST_model.pth", map_location=torch.device('cpu')))
except Exception as e:
    print(f"加载模型参数失败：{e}")
    print("请确保预测用的Net类和训练时完全一致！")
    exit()
model.eval()  # 关键：关闭dropout，进入预测模式
recent_preds = []
N = 5  # 平滑投票的帧数

# ===================== 3. 修正预处理（和训练时完全一致） =====================
# 训练时的预处理：Grayscale + Resize + ToTensor + Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 必须加，和训练时一致
])

# ===================== 4. 打开摄像头（修正索引为0） =====================
cap = cv2.VideoCapture(1)  # 默认摄像头是0，1为外接摄像头
if not cap.isOpened():
    print("摄像头打开失败，请检查驱动/摄像头是否被占用！")
    exit()

# 设置ROI区域（只识别中间200×200区域，减少背景干扰）
roi_top = 150
roi_bottom = 350
roi_left = 250
roi_right = 450

# ===================== 5. 实时识别循环（修正+优化） =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("读取摄像头帧失败！")
        break


    # 1. 提取ROI区域（只处理指定区域，减少背景干扰）
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    # 画ROI矩形框（方便用户对准）
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

    # 2. 图像预处理（和训练时对齐，关键！）
    # 灰度化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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

    # 4. 模型预测
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item()

    # 5. 投票平滑
    recent_preds.append(pred)
    if len(recent_preds) > N:
        recent_preds.pop(0)
    # 投票取最多的结果
    final_pred = max(set(recent_preds), key=recent_preds.count) if recent_preds else -1

    # 6. 显示结果（优化位置和样式）
    cv2.putText(
        frame,
        f'Prediction: {final_pred}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    # 显示预处理后的28×28图像（方便调试，看输入是否正确）
    img_show = cv2.resize(img, (100, 100))  # 放大显示
    frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = cv2.cvtColor((img_show*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # 7. 显示画面
    cv2.imshow('Handwriting Recognition (Press Q to Exit)', frame)

    # 按Q退出（英文输入法）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== 6. 释放资源 =====================
cap.release()
cv2.destroyAllWindows()