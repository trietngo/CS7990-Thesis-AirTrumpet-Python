import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

mp_hands = hands
hand = mp_hands.Hands()

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = drawing_styles

custom_style = mp_drawing_styles.get_default_hand_landmarks_style()
custom_connections = list(mp_hands.HAND_CONNECTIONS)

# Exclude the wrist, thumbs, and pinky fingers
excluded_landmarks = [
    HandLandmark.THUMB_CMC,
    HandLandmark.THUMB_IP,
    HandLandmark.THUMB_MCP,
    HandLandmark.THUMB_TIP,
    HandLandmark.PINKY_DIP,
    HandLandmark.PINKY_MCP,
    HandLandmark.PINKY_PIP,
    HandLandmark.PINKY_TIP,
    HandLandmark.WRIST
]

for landmark in excluded_landmarks:

    # we change the way the excluded landmarks are drawn
    custom_style[landmark] = DrawingSpec(color=(0,0,0), thickness=None)

    # we remove all connections which contain these landmarks
    custom_connections = [connection_tuple for connection_tuple in custom_connections 
                            if landmark.value not in connection_tuple]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Making predictions using hands model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image_rgb.flags.writeable = False
    results = hand.process(image_rgb)
    image_rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(hand_landmarks)
            mp_drawing.draw_landmarks(
                image_rgb, 
                hand_landmarks,
                custom_connections,
                custom_style
                # mp_hands.HAND_CONNECTIONS
            )

    # Convert the RGB image back to BGR
    image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    
    # Display the resulting frame
    cv.imshow('Hand Landmarks', image)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()