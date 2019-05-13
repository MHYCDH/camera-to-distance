import numpy as np
import cv2

KNOWN_WIDTH = 7.2834646  # 18.5cm   料斗长
KNOWN_HEIGHT = 2.8346457  # 7.2cm    料斗高


def find_marker(image):  # 检测边缘实现函数
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 颜色空间转换(转成灰度图)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波
    edged = cv2.Canny(gray, 35, 125)  # 边缘检测,输出二值图
    img, countours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 查找检测轮廓的轮廓
    c = max(countours, key=cv2.contourArea)  # 获取轮廓的(最大)面积的点集
    rect = cv2.minAreaRect(c)  # 返回轮廓的中心点坐标,
    return rect  # 表示rect[0]是最小外接矩形中心点坐标


# 标框
def show_frame(img_path):
    image = cv2.imread(img_path)
    cv2.imshow('image', image)

    marker = find_marker(image)
    box = cv2.boxPoints(marker)  # 得到坐标值(浮点型)
    box = np.int0(box)  # 将浮点型转换为整型
    cv2.drawContours(image, [box], -1, (255, 0, 255), 2)  # 在原图中描绘目标物体的轮廓(粉色)
    cv2.imshow('image', image)


if __name__ == "__main__":
    img_path = "LiaoDou2.jpg"

    while (1):
        show_frame(img_path)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()