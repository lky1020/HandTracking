import mediapipe as mp
import cv2
import csv
from os import path
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


def simpleGesture(handLandmarks):
    thumbIsOpen = False
    indexIsOpen = False
    middelIsOpen = False
    ringIsOpen = False
    pinkyIsOpen = False

    pseudoFixKeyPoint = handLandmarks[2].x
    if handLandmarks[3].x < pseudoFixKeyPoint and handLandmarks[4].x < pseudoFixKeyPoint:
        thumbIsOpen = True

    pseudoFixKeyPoint = handLandmarks[6].y
    if handLandmarks[7].y < pseudoFixKeyPoint and handLandmarks[8].y < pseudoFixKeyPoint:
        indexIsOpen = True

    pseudoFixKeyPoint = handLandmarks[10].y
    if handLandmarks[11].y < pseudoFixKeyPoint and handLandmarks[12].y < pseudoFixKeyPoint:
        middelIsOpen = True

    pseudoFixKeyPoint = handLandmarks[14].y
    if handLandmarks[15].y < pseudoFixKeyPoint and handLandmarks[16].y < pseudoFixKeyPoint:
        ringIsOpen = True

    pseudoFixKeyPoint = handLandmarks[18].y
    if handLandmarks[19].y < pseudoFixKeyPoint and handLandmarks[20].y < pseudoFixKeyPoint:
        pinkyIsOpen = True

    className = ""

    if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        className = "FIVE!"

    elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        className = "FOUR!"

    elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and not pinkyIsOpen:
        className = "THREE!"

    elif not thumbIsOpen and indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        className = "TWO!"

    elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        className = "ONE!"

    elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        className = "ROCK!"

    elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        className = "SPIDERMAN!"

    elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        className = "FIST!"

    return className

    # print(
    #     "FingerState: thumbIsOpen? " + str(thumbIsOpen) + " - indexIsOpen? " + str(indexIsOpen) + " - middelIsOpen? " +
    #     str(middelIsOpen) + " - ringIsOpen? " + str(ringIsOpen) + " - pinkyIsOpen? " + str(pinkyIsOpen))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(False, max_num_hands=2, min_detection_confidence=0.85, min_tracking_confidence=0.5)
hand_coords = 0

exportPicklePath = "HandGestureDataSet/" + "number.pkl"

with open(exportPicklePath, 'rb') as f:
    model = pickle.load(f)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Recolor Feed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Make Detections
    results = hands.process(img)

    # Recolor image back to BGR for rendering
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw Hand Landmarks
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            className = simpleGesture(handLms.landmark)

            # Grab Hand end coords
            hand_gesture_coords = tuple(np.multiply(
                np.array(
                    (handLms.landmark[mp_hands.HandLandmark.WRIST].x,
                     handLms.landmark[mp_hands.HandLandmark.WRIST].y)
                ), [640, 480]
            ).astype(int))

            cv2.putText(img, className, (hand_gesture_coords[0] - 75, hand_gesture_coords[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show Image
    cv2.imshow("Img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
