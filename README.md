# MNIST 手写数字识别项目

基于 PyTorch + CNN 的 MNIST 手写数字分类入门项目。
适合作为：

- 深度学习入门
- CNN 入门
- PyTorch 实践
- 计算机视觉基础项目
- 后续 OpenCV / 摄像头识别 的前置工程

项目从最基础的：

```text
Tensor → Dataset → DataLoader → 神经网络 → loss → backward
```

开始，逐步扩展到：

```text
CNN → 模型保存 → 模型推理 → 摄像头实时识别
```

非常适合零基础学习深度学习工程。

------

# 📖 项目介绍

本项目基于经典的 MNIST 手写数字数据集，实现：

- 手写数字分类
- CNN 卷积神经网络训练
- 模型参数保存与调用
- 图片推理预测
- OpenCV 摄像头测试
- 实时数字识别探索

项目重点不是“调库”，而是：

> 真正理解深度学习训练流程。

包括：

- Tensor 到底是什么
- 为什么需要 reshape
- forward 到底在干什么
- loss 如何计算
- optimizer 如何更新参数
- 模型参数到底存在哪里
- 推理阶段和训练阶段有什么区别

------

# 📊 数据集说明

本项目使用：

## MNIST 手写数字数据集

数据集特点：

- 0~9 共 10 类数字
- 28×28 灰度图
- 深度学习经典入门数据集

数据规模：

| 数据   | 数量     |
| ------ | -------- |
| 训练集 | 60000 张 |
| 测试集 | 10000 张 |

图片格式：

```text
[1, 28, 28]
```

含义：

```text
[channel, height, width]
```

------

# 🧠 项目实现内容

目前项目已经实现：

## ✅ MLP 全连接神经网络

基础结构：

```text
784 → 100 → 10
```

用于：

- 理解最基础神经网络结构
- 理解 Tensor shape
- 理解 forward 与 backward

------

## ✅ CNN 卷积神经网络

加入：

```python
nn.Conv2d()
nn.MaxPool2d()
```

实现：

- 图像特征提取
- 更高准确率
- CNN 基础学习

------

## ✅ 模型参数保存与加载

实现：

```python
torch.save()
torch.load()
```

支持：

- 模型持久化
- 增量训练
- 推理阶段调用

------

## ✅ 模型推理预测

实现：

```text
输入图片
→ 模型预测
→ 输出数字结果
```

示例：

```text
预测结果: 7
真实标签: 7
```

------

## ✅ OpenCV 摄像头测试

项目后期开始尝试：

- 摄像头实时采集
- 图像处理
- 实时数字识别

------

# 📂 项目结构

```text
MNIST/
│
├── CNN_MNIST_train.py          # CNN训练主程序
├── CNN_MNIST_fix.py            # CNN模型微调程序
├── CNN_MNIST_model.pth         # CNN模型参数
├── CNN_MNIST_modelpro.pth      # CNN增强版（微调后的）参数
│
├── Mnist_train.py              # MLP训练程序
├── Mnist_predict.py            # MLP推理程序
├── Mnist_model.pth             # MLP模型参数
│
├── MNIST采样脚本.py             # 数据采样脚本
├── 模型大规模测试.py             # 模型批量测试
│
├── 摄像头测试.py                # OpenCV摄像头测试
├── 摄像头优化.py                # 图像优化
│
├── CNN_MNIST前端模块.py         # 前端模块实验
├── 代码默写.py                  # 手搓代码练习本
│
├── data/MNIST/raw/             # 数据集
│
└── README.md
```

------

# ⚙️ 环境依赖

推荐环境：

```text
Python 3.10+
```

安装依赖：

```bash
pip install torch torchvision matplotlib opencv-python numpy
```

------

# 🚀 运行方式

------

# 1️⃣ 训练 MLP 模型

```bash
python Mnist_train.py
```

训练完成后：

```text
Mnist_model.pth
```

会自动生成。

------

# 2️⃣ 模型推理预测

```bash
python Mnist_predict.py
```

示例输出：

```text
预测结果: 7
真实标签: 7
```

------

# 3️⃣ 训练 CNN 模型

```bash
python CNN_MNIST_train.py
```

生成：

```text
CNN_MNIST_model.pth
```

------

# 4️⃣ 摄像头测试

```bash
python 摄像头测试.py
```

------

# 🔥 项目核心知识点

------

# Tensor 与 Shape

理解：

```text
[batch, channel, height, width]
```

以及：

```python
x.view(x.size(0), -1)
```

为什么需要 reshape。

------

# Dataset 与 DataLoader

明确：

## Dataset

定义：

> “一个样本是什么”

负责：

- 读取图片
- 读取标签

------

## DataLoader

定义：

> “训练时数据如何出现”

负责：

- batch
- shuffle
- 自动迭代

------

# forward 与 backward

理解：

## forward

```text
数据如何流动
```

## backward

```text
loss 如何反向传播梯度
```

------

# loss 与 optimizer

明确区分：

## criterion

定义：

```text
“如何衡量错误”
```

------

## loss

定义：

```text
一次前向传播后得到的结果
```

------

## optimizer

负责：

```text
根据梯度更新参数
```

------

# 🧩 建议学习路线（非常重要）

这个项目并不建议直接背代码（当然，对代码熟悉也是很重要的），建议按下面路线理解。

------

# 第一阶段：Python 与 PyTorch 基础

建议掌握：

- Python 类与对象，函数的定义与使用
- Tensor 基础
- shape
- view / reshape
```text
目标：看懂最基础的 PyTorch 代码，明白每一行在做什么
```

------

# 第二阶段：MLP 神经网络

建议掌握：

- nn.Module
- forward
- Linear
- ReLU
- loss
- optimizer
- backward
```text
目标：手搓完整 MNIST 训练代码
```

------

# 第三阶段：数据处理

建议掌握：

- Dataset
- DataLoader
- batch
- shuffle
- transforms
```text
目标：真正理解数据如何进入模型
```

------

# 第四阶段：CNN

建议掌握：

- Conv2d
- Pooling
- Feature Map
- 图像空间特征
```text
目标：理解 CNN 为什么适合图像任务
```

------

# 第五阶段：模型工程化

建议掌握：

- torch.save
- torch.load
- state_dict
- 推理阶段
- eval()
```text
目标：真正理解“模型”的存在形式
```

------

# 第六阶段：OpenCV 与实时识别

建议掌握：

- cv2.VideoCapture()
- 图像预处理
- 实时推理
```text
目标：完成摄像头实时数字识别
```

------

# 🎯 推荐学习方式

非常建议：

## 1️⃣ 手搓代码

不要复制粘贴。
比较有效的迭代学习方式是：

```text
自己默写
→ 报错
→ 修复
→ 再写
```

------

## 2️⃣ 理解 shape

建立对张量，向量，维度的认识

------

## 3️⃣ 不要急着学“大模型”

建议顺序：

```text
MLP
→ CNN
→ OpenCV
→ Transformer（attention）
→ Agent
```

------

# 📈 后续计划

项目未来准备继续扩展：

-  更深层 CNN
-  GPU 加速训练
-  GUI 界面
-  手写板输入
-  自定义数据集
-  摄像头实时识别优化
-  模型量化
-  ONNX 导出
-  Web 部署

------

# 💡 项目学习意义

有助于新手的入门，并学习到以下知识：

- 数据如何流动
- 参数如何更新
- 模型如何保存
- 模型如何被调用
- 推理阶段到底是什么

这是后续：

- 计算机视觉
- Transformer
- 大模型
- Agent
- 多模态

的基础。

------

# 👨‍💻 作者

RandiMingQiu (一个计算机小白)