import cv2

# 打开默认摄像头
cap = cv2.VideoCapture(1)

# 检查摄像头是否打开成功
if not cap.isOpened():
    print("摄像头打开失败，请检查驱动")
    exit()
print("按 q 键退出窗口")
while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        print("读取摄像头失败")
        break

    # 显示当前帧
    cv2.imshow('Camera Test', frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
