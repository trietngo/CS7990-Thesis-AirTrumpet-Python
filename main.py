import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Hands
mp_hands = hands
hand = mp_hands.Hands()

# Facial Recognition

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

            # print(len(hand_landmarks.landmark)) # should return 21 for 21 hand-knuckle coordinates

            # Curl Test

            # Index fingers
            index_pip_y = hand_landmarks.landmark[6].y
            index_tip_y = hand_landmarks.landmark[8].y

            # print("pip_y: " + str(hand_landmarks.landmark[6].y));
            # print("tip_y: " + str(hand_landmarks.landmark[8].y));

            if index_tip_y >= index_pip_y:
                print("Index finger curled")

                # Draw Text
                cv.putText(
                    image_rgb, # image on which to draw text
                    'Index finger curled', 
                    (200, 400), # bottom left corner of text
                    cv.FONT_HERSHEY_SIMPLEX, # font to use
                    0.5, # font scale
                    (255, 0, 0), # color
                    1, # line thickness
                )
            
            # Middle fingers
            middle_pip_y = hand_landmarks.landmark[10].y
            middle_tip_y = hand_landmarks.landmark[12].y

            # print("pip_y: " + str(hand_landmarks.landmark[6].y));
            # print("tip_y: " + str(hand_landmarks.landmark[8].y));

            if middle_tip_y >= middle_pip_y:
                print("Middle finger curled")

                # Draw Text
                cv.putText(
                    image_rgb, # image on which to draw text
                    'Middle finger curled', 
                    (200, 500), # bottom left corner of text
                    cv.FONT_HERSHEY_SIMPLEX, # font to use
                    0.5, # font scale
                    (255, 0, 0), # color
                    1, # line thickness
                )
            
            # Ring fingers
            ring_pip_y = hand_landmarks.landmark[14].y
            ring_tip_y = hand_landmarks.landmark[16].y

            # print("pip_y: " + str(hand_landmarks.landmark[6].y));
            # print("tip_y: " + str(hand_landmarks.landmark[8].y));

            if ring_tip_y >= ring_pip_y:
                print("Ring finger curled")

                # Draw Text
                cv.putText(
                    image_rgb, # image on which to draw text
                    'Ring finger curled', 
                    (200, 600), # bottom left corner of text
                    cv.FONT_HERSHEY_SIMPLEX, # font to use
                    0.5, # font scale
                    (255, 0, 0), # color
                    1, # line thickness
                )
            
            print()

            # print("x: " + str(hand_landmarks.landmark[5].x));
            # print("y: " + str(hand_landmarks.landmark[5].y));
            # print("z: " + str(hand_landmarks.landmark[5].z));

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