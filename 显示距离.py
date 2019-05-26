def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray",gray)
    # cv2.waitKey(0)
    classifier = cv2.CascadeClassifier("D:\Programer\JetBrains\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml")
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

import cv2

cap = cv2.VideoCapture(0) #参数0表示获取第一个摄像头
#对视频流进行逐帧显示
while(1): #对视频不断进行采集显示
    ret,img = cap.read() #读取帧数据
    face_detect(img)
    cv2.imshow("Image",img) #对帧图片进行显示
    #当用户输入q时，退出采集显示过程
    if cv2.waitKey(1) & 0xff == ord("q"): #判断用户是否要退出
        break
cap.release() #释放摄像头
cv2.destroyAllWindows() #关闭所有图片窗口

