import cv2
import time
import os
import HandTrackingModule as htm
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import keyboard as ky


def Case(i):
    if i == 'f':
        FingerCounting()

    if i == "v":
        VolumeControl()


def FingerCounting():
    # Finger Counting
    fingers = []

    # thumb
    if lmList[5][1] > lmList[17][1]:
        # Left
        # [index finger][height]
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][2] + 100:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        # Right
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
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    totalFingers = fingers.count(1)
    print(totalFingers)

    h, w, c = overlayList[totalFingers - 1].shape
    img[0:h, 0:w] = overlayList[totalFingers - 1]


def VolumeControl():
    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
    cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)

    # Hand Range 35 - 275
    # volume range -65 - 0

    vol = np.interp(length, [35, 275], [minVol, maxVol])
    volBar = np.interp(length, [35, 275], [400, 150])
    volPer = np.interp(length, [35, 275], [0, 100])
    print(vol)

    volume.SetMasterVolumeLevel(vol, None)

    if length < 35:
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, F'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


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

# Finger Counting
tipIds = [4, 8, 12, 16, 20]

# Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

key = ''

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if ky.is_pressed('f'):
        key = 'f'

    elif ky.is_pressed('v'):
        key = 'v'

    if len(lmList) != 0 and key != '':
        Case(key)

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
