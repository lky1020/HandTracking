import cv2
import os
import mediapipe as mp

# Create object for mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # used for drawing the hand landmarks

for x in range(1, 17):
    directory = "yes/"
    imgName = "yes (" + str(x) +").jpg"
    path = directory + imgName

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # Read the img from specified path
    img = cv2.imread(path)

    # convert img to rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # process the img with mediapipe
    results = hands.process(imgRGB)

    # ensure system detect hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    name = './data/' + imgName
    print('Creating...' + name)

    # writing the extracted images
    cv2.imwrite(name, img)

    # Release all space and windows once done
    cv2.destroyAllWindows()
