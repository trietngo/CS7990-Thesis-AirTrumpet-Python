import numpy as np
import pyaudio
import wave
import cv2 as cv
import mediapipe as mp
import math
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# # Playing sound using PyAudio
# chunk = 1024

# # Open wav file
# f = wave.open(r"data/sample_sounds/357382__mtg__trumpet-c4.wav", "rb")

# # Init PyAudio object
# p = pyaudio.PyAudio()

# # Open stream
# stream = p.open(
#     format=p.get_format_from_width(f.getsampwidth()),
#     channels=f.getnchannels(),
#     rate=f.getframerate(),
#     output=True
# )

# # Read data
# data = f.readframes(chunk)

# # Play stream
# while data:
#     stream.write(data)
#     data = f.readframes(chunk)

# # Stop stream
# stream.stop_stream()
# stream.close()

# # Close pyAudio
# p.terminate()

# FACIAL RECOGNITION
mp_face = face_mesh
face = mp_face.FaceMesh()

# HAND GESTURE RECOGNITION
mp_hands = hands
hand = mp_hands.Hands()

# Initialize drawing utils for the hand and facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_lips = mp.solutions.drawing_utils

mp_drawing_styles = drawing_styles

custom_style_hands = mp_drawing_styles.get_default_hand_landmarks_style()
custom_connections_hands = list(mp_hands.HAND_CONNECTIONS)

custom_style_lips = mp_drawing_styles.get_default_face_mesh_tesselation_style()
custom_connections_lips = list(mp_face.FACEMESH_LIPS)

# Extract all lip landmarks
lip_landmarks_2d = [list(connection_tuple) for connection_tuple in custom_connections_lips]

# Flatten the lip landmarks
lip_landmarks = [
    lip_landmark
    for pair in lip_landmarks_2d
    for lip_landmark in pair
]

# Remove duplicates
lip_landmarks = list(set(lip_landmarks))

# # Remove non-lip feature drawing
# for lip_landmark_tuple in custom_connections_lips:
#     custom_style_lips[lip_landmark_tuple] = DrawingSpec(color=(0,0,0), thickness=None)
#     print(custom_style_lips[lip_landmark_tuple])

# for style in custom_style_lips:
#     print(style)

# Exclude the wrist, thumbs, and pinky fingers
excluded_landmarks_hands = [
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

for hand_landmark in excluded_landmarks_hands:

    # we change the way the excluded landmarks are drawn
    custom_style_hands[hand_landmark] = DrawingSpec(color=(0,0,0), thickness=None)

    # we remove all connections which contain these landmarks
    custom_connections_hands = [
        connection_tuple
        for connection_tuple in custom_connections_hands 
        if hand_landmark.value not in connection_tuple
    ]

print("Custom style hands: ")
print(custom_style_hands)

# print("Custom connection hands: ")
# print(custom_connections_hands)

print("Custom style lips: ")
print(custom_style_lips)

# print("Custom connection lips after fix: ")
# print(custom_connections_lips)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def airflowClassification(

    # Lip Landmarks
    lip_center_outer_upper_y,
    lip_center_outer_lower_y,

    lip_center_inner_upper_y,
    lip_center_inner_lower_y,

    lip_left_x,
    lip_right_x
):
    
    y_diff_outer_center_lip = abs(lip_center_outer_upper_y - lip_center_outer_lower_y)

    y_diff_inner_center_lip = math.ceil(lip_center_inner_upper_y) >= math.floor(lip_center_inner_lower_y)

    x_diff_edge_lip = abs(lip_left_x - lip_right_x)

    # print("lip_center_outer_upper_y: ", lip_center_outer_upper_y)
    # print("lip_center_outer_lower_y: ", lip_center_outer_lower_y)
    # print()
    print("diff: ", abs(lip_center_outer_upper_y - lip_center_outer_lower_y))

    print("lip_left_x: ", lip_left_x)
    print("lip_right_x: ", lip_right_x)

    print("x_diff: ", x_diff_edge_lip)

    # Closed lips
    if y_diff_inner_center_lip:

        # Pursed lips, more air
        # 0.05 for mac
        if y_diff_outer_center_lip <= 0.05:
            return "Strained"
        
        elif x_diff_edge_lip <= 0.1:
            return "Pursed"
        
        else:
            return "Closed"

    else:
        return "Open"

def valvesClassification(
        
    # Hand Landmarks
    index_pip_y,
    index_tip_y,
    middle_pip_y,
    middle_tip_y,
    ring_pip_y,
    ring_tip_y,

):

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

    valve_state = "None"

    if isBackValvePressed() or isMiddleValvePressed() or isFrontValvePressed():
        valve_state = ""

    # No valve pressed
    if isBackValvePressed():
        valve_state += "Back"
    
    # Back valve pressed only
    if isMiddleValvePressed():
        valve_state += "Middle"
    
    # Middle valve pressed only
    if isFrontValvePressed():
        valve_state += "Front"
    
    return valve_state

# Finger pos
# "#" means sharp

# Closed
# Strained
# Pursed
# Strained and Pursed

# C4: None + Closed
# C#4: All + Closed
# D4: BackFront + Closed
# D#4: MiddleFront + Closed
# E4: BackMiddle + Closed
# F4: Back + Closed
# F#4: Middle + Closed

# G4: None + Pursed
# G#4: MiddleFront + Pursed
# A4: BackMiddle + Pursed
# A#4: Back + Pursed
# B4: Middle + Pursed

# C5: None + Strained
# C#5: All + Closed
# D5: BackFront + Closed
# D#5: MiddleFront + Closed
# E5: BackMiddle + Closed
# F5: Back + Closed
# F#5: Middle + Closed

# G5: None + Pursed
# G#5: MiddleFront + Pursed
# A5: BackMiddle + Pursed
# A#5: Back + Pursed
# B5: Middle + Pursed

# C6: None + Strained


def notesClassification(valve_state, lip_state):
    
    if lip_state == "Closed":
        
        if valve_state == "None":
            return "C4 - Do"
        
        elif valve_state == "BackFront":
            return "D4 - Re"
        
        elif valve_state == "BackMiddle":
            return "E4 - Mi"
    
    elif lip_state == "Pursed":
        if valve_state == "None":
            return "C5"
    
    return "Nothing is playing"

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
    results_hands = hand.process(image_rgb)
    results_face = face.process(image_rgb)
    image_rgb.flags.writeable = True

    valve_state = "Not Detected"
    lip_state = "Not Detected"

    if results_hands.multi_hand_landmarks:

        for hand_landmarks in results_hands.multi_hand_landmarks:

            valve_state = valvesClassification(
                
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

            # Draw Text
            cv.putText(
                image_rgb, # image on which to draw text
                valve_state, 
                (200, 400), # bottom left corner of text
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                1, # font scale
                (255, 0, 0), # color
                2, # line thickness
            )

            mp_drawing.draw_landmarks(
                image_rgb, 
                hand_landmarks,
                custom_connections_hands,
                custom_style_hands
                # mp_hands.HAND_CONNECTIONS
            )
    
    if results_face.multi_face_landmarks:

        for face_landmark in results_face.multi_face_landmarks:

            lip_state = airflowClassification(

                # Outer boundary of lips
                lip_center_outer_upper_y=face_landmark.landmark[0].y,
                lip_center_outer_lower_y=face_landmark.landmark[17].y,

                # Inner boundary of lips
                lip_center_inner_upper_y=face_landmark.landmark[13].y,
                lip_center_inner_lower_y=face_landmark.landmark[14].y,

                # Commmissures
                lip_left_x=face_landmark.landmark[291].x,
                lip_right_x=face_landmark.landmark[61].x
            )

            # Draw Text
            cv.putText(
                image_rgb, # image to draw text on
                lip_state, 
                (200, 450), # bottom left corner of text
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                1, # font scale
                (255, 0, 0), # color
                2, # line thickness
            )

            mp_drawing_lips.draw_landmarks(
                image_rgb, 
                face_landmark,
                # connections=custom_connections_lips,
                landmark_drawing_spec=custom_style_lips
                # mp_face.FACEMESH_CONTOURS
            )

    if results_hands.multi_hand_landmarks and results_face.multi_face_landmarks:

        notes_output = notesClassification(valve_state, lip_state)

        cv.putText(
            image_rgb, # image to draw text on
            "Current note: " + notes_output, 
            (200, 500), # bottom left corner of text
            cv.FONT_HERSHEY_SIMPLEX, # font to use
            1, # font scale
            (255, 0, 0), # color
            2, # line thickness
        )

    # Convert the RGB image back to BGR
    image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    
    # Display the resulting frame
    cv.imshow('Hand Landmarks', image)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()