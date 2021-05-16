import mediapipe as mp
import cv2
import csv
from os import path
import numpy as np
import pandas as pd
import pickle


def exportCsv(result):
    num_coords = len(result.multi_hand_landmarks[0].landmark)

    landmarks = ['class']

    for val in range(1, num_coords + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

    if not path.exists('coords.csv'):
        with open('coords.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

    class_name = "Hello"

    # Export Coordinates
    try:
        # Extract Hand Landmarks
        hands_coords = result.multi_hand_landmarks[0].landmark
        hands_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hands_coords]).flatten())

        row = hands_row
        row.insert(0, class_name)

        print(row)

        # with open('coords.csv', mode='a', newline='') as f:
        #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(row)

    except:
        pass


def handGestureDetection(result):


        # Export Coordinates
        try:
            # Extract Hand Landmarks
            hands_coords = result.multi_hand_landmarks[0].landmark
            hands_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hands_coords]).flatten())

            row = hands_row

            # Make Detections
            x = pd.DataFrame([row])
            hand_gesture_class = model.predict(x)[0]
            hand_gesture_prob = model.predict_proba(x)[0]
            print(hand_gesture_class, hand_gesture_prob)

            # Grab Hand end coords
            hand_gesture_coords = tuple(np.multiply(
                np.array(
                    (result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x, result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y)
                ), [640, 480]
            ).astype(int))


        except Exception as e:
            print(e)
            pass

        return hand_gesture_class, hand_gesture_coords


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(False, max_num_hands=2, min_detection_confidence=0.85, min_tracking_confidence=0.5)
hand_coords = 0

with open('body_language.pkl', 'rb') as f:
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

            try:
                # Extract Hand Landmarks
                hands_coords = handLms.landmark
                hands_row = list(
                    np.array([[landmark.x, landmark.y, landmark.z] for landmark in hands_coords]).flatten())

                row = hands_row

                # Make Detections
                x = pd.DataFrame([row])
                hand_gesture_class = model.predict(x)[0]
                hand_gesture_prob = model.predict_proba(x)[0]
                print(hand_gesture_class, hand_gesture_prob)

                # Grab Hand end coords
                hand_gesture_coords = tuple(np.multiply(
                    np.array(
                        (handLms.landmark[mp_hands.HandLandmark.WRIST].x,
                         handLms.landmark[mp_hands.HandLandmark.WRIST].y)
                    ), [640, 480]
                ).astype(int))

                # cv2.rectangle(img, (hand_gesture_coords[0] - 50, hand_gesture_coords[1] + 40),
                #               ((hand_gesture_coords[0] + len(hand_gesture_class) * 20) - 50, hand_gesture_coords[1]), (245, 117, 16), -1)

                cv2.putText(img, hand_gesture_class, (hand_gesture_coords[0] - 50, hand_gesture_coords[1] + 30)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

    # Show Image
    cv2.imshow("Img", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
