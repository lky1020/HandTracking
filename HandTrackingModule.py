import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Create object for mediapipe hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # used for drawing the hand landmarks

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # convert img to rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process the img with mediapipe
        self.results = self.hands.process(imgRGB)

        # ensure system detect hand
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    # draw the hand landmarks
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        xList = []
        yList = []
        bbox = []

        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # get the position of each landmarks
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                xList.append(cx)
                yList.append(cy)

                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Bounding Box
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bbox = xMin, yMin, xMax, yMax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []

        # thumb
        if self.lmList[5][1] > self.lmList[17][1]:
            #  Right
            # [index finger][height]
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][2] + 100:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Left
            # [index finger][height]
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            # [index finger][height]
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    cap = cv2.VideoCapture(0)

    # Used to calculate FPS
    prevTime = 0
    currentTime = 0

    # create object
    detector = handDetector()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            print(lmList[0])

        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
