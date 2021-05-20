# Project 1 - Virtual Paint
import cv2
import numpy as np

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

myColors = [[115, 77, 197, 168, 255, 255],
            [25, 20, 160, 40, 255, 255],
            [165, 35, 195, 179, 255, 255]]

# BGR
myColorsValues = [[124, 51, 198],
                  [152, 226, 217],
                  [126, 70, 223]]

myPoints = []  # x, y, colorId


def findColor(img, myColors, myColorsValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []

    # get the mask of the image based on the lower and upper
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)

        # Draw circle around the color
        cv2.circle(imgResult, (x, y), 15, myColorsValues[count], cv2.FILLED)

        if x != 0 and y != 0:
            newPoints.append([x, y, count])
            print(count)

        count += 1

    return newPoints

def getContours(img):
    # Get the detail of contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x, y, w, h = 0, 0, 0, 0

    # Handle the contours
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Calculate the area

        if area > 500:
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)  # Draw the contours to the imgContours
            peri = cv2.arcLength(cnt, True)  # Find arc length of cnt
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # detect the corner of the cnt
            x, y, w, h = cv2.boundingRect(approx)  # get x, y, width, height

    return x + (w // 2), y

def drawOnCancas(myPoints, myColorsValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorsValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorsValues)

    if len(newPoints) != 0:
        for point in newPoints:
            myPoints.append(point)

    if len(myPoints) != 0:
        drawOnCancas(myPoints, myColorsValues)

    cv2.imshow("Result", imgResult)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
