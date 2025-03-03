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

# Playing sound using PyAudio
chunk = 512

# Open wav file
f = wave.open(r"data\sample_sounds\357382__mtg__trumpet-c4.wav", "rb")

# Init PyAudio object
p = pyaudio.PyAudio()

# Open stream
stream = p.open(
    format=p.get_format_from_width(f.getsampwidth()),
    channels=f.getnchannels(),
    rate=f.getframerate(),
    output=True
)

# Read data
data = f.readframes(chunk)

# Play stream
while data:
    stream.write(data)
    data = f.readframes(chunk)

# Stop stream
stream.stop_stream()
stream.close()

# Close pyAudio
p.terminate()

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

def notesClassification(
        
    # Hand Landmarks
    index_pip_y,
    index_tip_y,
    middle_pip_y,
    middle_tip_y,
    ring_pip_y,
    ring_tip_y,

    # Lip Landmarks
    # lip_center_outer_upper_y,

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

    # No valve pressed
    if not isBackValvePressed() and not isMiddleValvePressed() and not isFrontValvePressed():
        return "No valves pressed. Possible notes: C"
    
    # Back valve pressed only
    elif isBackValvePressed() and not isMiddleValvePressed() and not isFrontValvePressed():
        return "Back valve pressed. "
    
    # Middle valve pressed only
    elif not isBackValvePressed() and isMiddleValvePressed() and not isFrontValvePressed():
        return "Middle valve pressed"
    
    # Front valve pressed only
    elif not isBackValvePressed() and not isMiddleValvePressed() and isFrontValvePressed():
        return "Front valve pressed"
    
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
    results_hands = hand.process(image_rgb)
    results_face = face.process(image_rgb)
    image_rgb.flags.writeable = True

    if results_hands.multi_hand_landmarks:

        for hand_landmarks in results_hands.multi_hand_landmarks:

            output_hands = notesClassification(
                
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
                output_hands, 
                (200, 400), # bottom left corner of text
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                0.5, # font scale
                (255, 0, 0), # color
                1, # line thickness
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

            # print(face_landmark)

            lip_center_outer_upper_y = face_landmark.landmark[0].y
            lip_center_outer_lower_y = face_landmark.landmark[17].y

            # print("lip_center_outer_upper_y: " + str(lip_center_outer_upper_y))
            # print("lip_center_outer_lower_y: " + str(lip_center_outer_lower_y))

            lip_center_inner_upper_y = face_landmark.landmark[13].y
            lip_center_inner_lower_y = face_landmark.landmark[14].y

            output_lips = ""

            # Closed lips
            if math.ceil(lip_center_inner_upper_y * 100) >= math.floor(lip_center_inner_lower_y * 100):
                output_lips += "Mouth closed "

                # Pursed lips, more air
                if abs(lip_center_outer_upper_y - lip_center_outer_lower_y) <= 0.03:

                    output_lips += "Mouth pursed "

            print(output_lips)

            # Draw Text
            cv.putText(
                image_rgb, # image on which to draw text
                output_lips, 
                (200, 425), # bottom left corner of text
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                0.5, # font scale
                (255, 0, 0), # color
                1, # line thickness
            )

            mp_drawing_lips.draw_landmarks(
                image_rgb, 
                face_landmark,
                # connections=custom_connections_lips,
                landmark_drawing_spec=custom_style_lips
                # mp_face.FACEMESH_CONTOURS
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