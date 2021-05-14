import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "VirtualPaintIcon"
myList = os.listdir(folderPath)
print(myList)

overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

header = overlayList[0]
paintColor = (0, 0, 255)
brushThickness = 15
eraserThickness = 50

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

        # If Selection Mode - Two Finger are Up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('Selection Mode')

            # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    paintColor = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    paintColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    paintColor = (0, 255, 0)
                if 1050 < x1 < 1200:
                    header = overlayList[3]
                    paintColor = (0, 0, 0)

            cv2.circle(img, (x1, y1), 15, paintColor, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, paintColor, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), paintColor, 3)

        # If Drawing Mode - Index Finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, paintColor, cv2.FILLED)
            print('Drawing Mode')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if paintColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), paintColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), paintColor, eraserThickness)

            else:
               cv2.line(img, (xp, yp), (x1, y1), paintColor, brushThickness)
               cv2.line(imgCanvas, (xp, yp), (x1, y1), paintColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # draw black
    img = cv2.bitwise_and(img, imgInv)

    # Get Color for paint
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the Header Image
    img[0: 125, 0: 1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
