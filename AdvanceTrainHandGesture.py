import mediapipe as mp
import cv2
import copy
import itertools
import csv
import time
import numpy as np
from os import path


def main():
    # FPS Preparation
    prevTime = 0
    currentTime = 0

    # Camera Preparation
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Model Preparation
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.5)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Recolor Feed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Make Detections
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True

        # Recolor image back to BGR for rendering
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw Hand Landmarks
        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate Bounding Box
                bbox = calc_bounding_rect(img, hand_landmarks)

                # Calculate Landmarks
                landmark_list = calc_landmark_list(img, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Write to the dataset file
                print(pre_processed_landmark_list)
                className = "4"
                logging_csv(className, False, pre_processed_landmark_list)

                # Drawing
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                img = draw_bounding_rect(True, img, bbox)

        # Calculate Fps
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow("Img", img)

        key = cv2.waitKey(10)
        if key == 27:
            break


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(className, mode, landmark_list):
    csv_path = "HandGestureDataSet/model/number.csv"

    if mode:
        if not path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow("")

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([className, *landmark_list])

    return


def draw_bounding_rect(draw, img, bbox):
    if draw:
        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

    return img


if __name__ == "__main__":
    main()
