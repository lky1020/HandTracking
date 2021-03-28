import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)

# Used to calculate FPS
prevTime = 0
currentTime = 0

# create object
detector = htm.handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #
    # if len(lmList) != 0:
    #     print(lmList[0])

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)