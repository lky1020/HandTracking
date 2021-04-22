import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# store images
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')

    overlayList.append(image)

prevTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # thumb
        if lmList[5][1] > lmList[17][1]:
            #  Right
            # [index finger][height]
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][2] + 100:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Left
            # [index finger][height]
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        print("Index 4 = " + str(lmList[tipIds[0]][1]))
        print("Index 3 = " + str(lmList[tipIds[0] - 1][2]))

        # 4 fingers
        for id in range(1, 5):
            # [index finger][height]
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
