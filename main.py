import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Hands
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

def notesClassification(index_pip_y, index_tip_y, middle_pip_y, middle_tip_y, ring_pip_y, ring_tip_y):

    # Function to check if the front valve is pressed
    # if the y-coord of the index tip is lower than the y-coord of the index pip
    # then register a press and return True
    def isBackValvePressed():

        # Use greater than because the coords are inverted
        if index_tip_y >= index_pip_y:
            return True
        
        else:
            return False

    # Function to check if the middle valve is pressed
    # if the y-coord of the middle finger tip is lower than the y-coord of the middle finger pip
    # then register a press and return True
    def isMiddleValvePressed():

        # Use greater than because the coords are inverted
        if middle_tip_y >= middle_pip_y:
            return True
        
        else:
            return False

    # Function to check if the back valve is pressed
    # if the y-coord of the ring finger tip is lower than the y-coord of the ring finger pip
    # then register a press and return True
    def isFrontValvePressed():

        # Use greater than because the coords are inverted
        if ring_tip_y >= ring_pip_y:
            return True
        
        else:
            return False
    
    # PRESSED VALVE COMBINATIONS
    
    # backValvePressed = isBackValvePressed()
    # middleValvePressed = isMiddleValvePressed()
    # frontValvePressed = isFrontValvePressed()

    # print("back: " + str(backValvePressed))
    # print("middle: " + str(middleValvePressed))
    # print("front: " + str(frontValvePressed))

    # No valve pressed
    if not isBackValvePressed() and not isMiddleValvePressed() and not isFrontValvePressed():
        return "No valves pressed. Possible notes: Do"
    
    # Back valve pressed only
    elif isBackValvePressed() and not isMiddleValvePressed() and not isFrontValvePressed():
        return "Back valve pressed. "
    
    # Middle valve pressed only
    elif not isBackValvePressed() and isMiddleValvePressed() and not isFrontValvePressed():
        return "Middle valve pressed"
    
    # Front valve pressed only
    elif not isBackValvePressed() and not isMiddleValvePressed() and isFrontValvePressed():
        return "Front valved pressed"
    
    # Back and middle valves only
    elif isBackValvePressed() and isMiddleValvePressed() and not isFrontValvePressed():
        return "Back and middle valves pressed"

    # Middle and front valves only
    elif not isBackValvePressed() and isMiddleValvePressed() and isFrontValvePressed():
        return "Middle and front valves pressed"

    # Back and front valves only
    elif isBackValvePressed() and not isMiddleValvePressed() and isFrontValvePressed():
        return "Back and front valves pressed"

    # All three valves pressed
    elif isBackValvePressed() and isMiddleValvePressed() and isFrontValvePressed():
        return "All 3 valves pressed"


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

            output = notesClassification(
                
                # Index finger
                index_pip_y=hand_landmarks.landmark[6].y,
                index_tip_y=hand_landmarks.landmark[8].y,

                # Middle finger
                middle_pip_y=hand_landmarks.landmark[10].y,
                middle_tip_y=hand_landmarks.landmark[12].y,

                # Ring finger
                ring_pip_y=hand_landmarks.landmark[14].y,
                ring_tip_y=hand_landmarks.landmark[16].y
            )
            
            print()

            # Draw Text
            cv.putText(
                image_rgb, # image on which to draw text
                output, 
                (200, 400), # bottom left corner of text
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                0.5, # font scale
                (255, 0, 0), # color
                1, # line thickness
            )

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