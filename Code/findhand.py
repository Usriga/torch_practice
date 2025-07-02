import cv2
import mediapipe as mp
import argparse
import time

ap = argparse.ArgumentParser()

ap.add_argument("--mode", default=False, help="Process Image or Video")
ap.add_argument("--maxHands", default=2, help="Maximum number of hands to detect")
ap.add_argument("--model_complexity", default=1, help="Hand landmark model complexity: 0 or 1")
ap.add_argument("--detectionCon", default=0.8, help="Minimum detection confidence threshold")
ap.add_argument("--trackCon", default=0.8, help="Minimum tracking confidence threshold")
ap.add_argument("--wCam",default=640, help="Width of Camera")
ap.add_argument("--hCam",default=480, help="Height of Camera")

args = ap.parse_args()

cap = cv2.VideoCapture(0)
cap.set(3,args.wCam) 
cap.set(4,args.hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(args.mode,args.maxHands,args.model_complexity,args.detectionCon,args.trackCon)
mpDraw = mp.solutions.drawing_utils
results = 0

def findHands(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    global results
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img

# 获取关节点位置
def findPosition(img, draw=True):
    lmLists = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmLists.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)

    return lmLists

def main():
    pTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(3,args.wCam) #宽度
    cap.set(4,args.hCam) #高度
    
    if  cap.isOpened():
        open,frame = cap.read()
    else:
        open = False
    
    while open:
        success,img = cap.read()
        img = findHands(img)
        lmList = findPosition(img,draw = True) # 画不画的完全没必要

        ### 按关节相对位置进行手势的识别，完全不够健壮，此处可以将位置点和真实标签做分类学习，加入判断
        if len(lmList) != 0:
            max_list = [lmList[4][2], lmList[8][2], lmList[12][2], lmList[16][2], lmList[20][2]]  # 每个手指的尖端部位

            count = 0  # 手势数字结果

            # 手势为4
            if max_list[1] < lmList[9][2] and max_list[2] < lmList[9][2] and max_list[3] < lmList[9][2] and max_list[
                4] < \
                    lmList[9][2] and max_list[0] > lmList[9][2] and max_list[0] > lmList[17][2]:
                count = 4
            # 手势为3
            elif max_list[1] < lmList[9][2] and max_list[2] < lmList[9][2] and max_list[3] < lmList[9][2] and \
                    lmList[20][
                        2] > lmList[9][2]:
                count = 3
            # 手势为2
            elif max_list[1] < lmList[9][2] < lmList[16][2] and max_list[2] < lmList[9][2] < lmList[20][2]:
                count = 2
            # 手势为1
            elif max_list[1] < lmList[9][2] < lmList[16][2] and lmList[20][2] > lmList[9][2] and lmList[12][2] > \
                    lmList[9][
                        2]:
                count = 1
            # 手势为5
            else:
                count = 5

            HandImage = cv2.imread(f'D:/pytorch_practice/Dataset/findhandimage/{count}.jpg')
            HandImage = cv2.resize(HandImage, (150, 200))
            h, w, c = HandImage.shape
            img[0:h, 0:w] = HandImage  # 将视频左上角覆盖手势图片
            cv2.putText(img, f'{int(count)}', (400, 280), cv2.FONT_HERSHEY_PLAIN, 20, (255, 0, 255), 10)  # 显示手势图片

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # 每秒传输帧数
        pTime = cTime
        cv2.putText(img, f'fps: {int(fps)}', (500, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # 将帧数显示在窗口
        cv2.imshow("Image", cv2.resize(img, (args.wCam, args.hCam)))
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    main()