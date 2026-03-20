import cv2
import os
import time

# 配置部分
save_root = "MNIST_Pro_digits"
digits = [str(i) for i in range(10)]
img_size = 28                   # resize 后的尺寸

# 创建数字文件夹
for d in digits:
    os.makedirs(os.path.join(save_root, d), exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("摄像头打开失败！")
    exit()

print("摄像头已打开，按数字键 0~9 拍摄，按 q 退出")


# 拍摄循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 显示原始画面
    cv2.imshow("Capture Digits", frame)

    key = cv2.waitKey(1) & 0xFF

    # 按数字键拍摄
    if key >= ord('0') and key <= ord('9'):
        digit = chr(key)
    #灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        resized = cv2.resize(binary, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # 文件名加时间戳避免重复
        filename = f"{int(time.time()*1000)}.png"
        save_path = os.path.join(save_root, digit, filename)
        cv2.imwrite(save_path, resized)
        print(f"保存 {digit}: {filename}")

    # 按 q 退出
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
