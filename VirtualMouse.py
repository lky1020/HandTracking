import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
# Frame Reduction
frameR = 100
smoothening = 7

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
prevLocX, prevLocY = 0, 0
curLocX, curLocY = 0, 0

detector = htm.handDetector(detectionCon=0.85, maxHands=1)

while True:
    # Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Draw Bounding Box for mouse moving
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 0, 255), 3)

    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

        # Both Index and Middle Fingers are up: Moving Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen Values
            curLocX = prevLocX + (x3 - prevLocX) / smoothening
            curLocY = prevLocY + (y3 - prevLocY) / smoothening

            # Move Mouse
            autopy.mouse.move(wScr - curLocX, curLocY)
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

            prevLocX, prevLocY = curLocX, curLocY

        # Only Index Finger: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 0:
            autopy.mouse.click()

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display Result
    cv2.imshow("Image", img)
    cv2.waitKey(1)
