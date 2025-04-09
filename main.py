import numpy as np
import pyaudio
import wave
import enum
from time import sleep
from pygame import mixer
import cv2 as cv
import mediapipe as mp
import math
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Distance from face to webcam is 40cm
STANDARD_DISTANCE = 40

# Enumerate all possible Trumpet Notes
# And the corresponding audio files in the data folder
class TrumpetNoteSample(enum.StrEnum):

    """Audio Files of 24 Possible Trumpet Notes"""

    C_4 = "data/sample_sounds_truncated/357382__mtg__trumpet-c4-truncated.wav"
    C_SHARP_4 = "data/sample_sounds_truncated/357478__mtg__trumpet-csharp4-truncated.wav"
    D_4 = "data/sample_sounds_truncated/357542__mtg__trumpet-d4-truncated.wav"
    D_SHARP_4 = "data/sample_sounds_truncated/357388__mtg__trumpet-dsharp4-truncated.wav"
    E_4 = "data/sample_sounds_truncated/357544__mtg__trumpet-e4-truncated.wav"
    F_4 = "data/sample_sounds_truncated/357384__mtg__trumpet-f4-truncated.wav"
    F_SHARP_4 = "data/sample_sounds_truncated/357323__mtg__trumpet-fsharp4-truncated.wav"
    G_4 = "data/sample_sounds_truncated/357360__mtg__trumpet-g4-truncated.wav"
    G_SHARP_4 = "data/sample_sounds_truncated/357363__mtg__trumpet-gsharp4-truncated.wav"
    A_4 = "data/sample_sounds_truncated/357370__mtg__trumpet-a4-truncated.wav"
    A_SHARP_4 = "data/sample_sounds_truncated/357457__mtg__trumpet-asharp4-truncated.wav"
    B_4 = "data/sample_sounds_truncated/247067__mtg__overall-quality-of-single-note-trumpet-b4-truncated.wav"

    C_5 = "data/sample_sounds_truncated/357432__mtg__trumpet-c5-truncated.wav"
    C_SHARP_5 = "data/sample_sounds_truncated/357385__mtg__trumpet-csharp5-truncated.wav"
    D_5 = "data/sample_sounds_truncated/357336__mtg__trumpet-d5-truncated.wav"
    D_SHARP_5 = "data/sample_sounds_truncated/357386__mtg__trumpet-dsharp5-truncated.wav"
    E_5 = "data/sample_sounds_truncated/357351__mtg__trumpet-e5-truncated.wav"
    F_5 = "data/sample_sounds_truncated/357546__mtg__trumpet-f5-truncated.wav"
    F_SHARP_5 = "data/sample_sounds_truncated/357361__mtg__trumpet-fsharp5-truncated.wav"
    G_5 = "data/sample_sounds_truncated/357364__mtg__trumpet-g5-truncated.wav"
    G_SHARP_5 = "data/sample_sounds_truncated/357433__mtg__trumpet-gsharp5-truncated.wav"
    A_5 = "data/sample_sounds_truncated/357328__mtg__trumpet-a5-truncated.wav"
    A_SHARP_5 = "data/sample_sounds_truncated/357469__mtg__trumpet-asharp5-truncated.wav"
    # B_5 = ""

# Playing sound using PyAudio
def playAudio(path):

    mixer.music.load(path)
    mixer.music.set_volume(1)
    mixer.music.play(loops=-1, start=0)

def playNote(note):

    match note:
        case "C4":
            playAudio(TrumpetNoteSample.C_4)
        case "C#4":
            playAudio(TrumpetNoteSample.C_SHARP_4)
        case "D4":
            playAudio(TrumpetNoteSample.D_4)
        case "D#4":
            playAudio(TrumpetNoteSample.D_SHARP_4)
        case "E4":
            playAudio(TrumpetNoteSample.E_4)
        case "F4":
            playAudio(TrumpetNoteSample.F_4)
        case "F#4":
            playAudio(TrumpetNoteSample.F_SHARP_4)
        case "G4":
            playAudio(TrumpetNoteSample.G_4)
        case "G#4":
            playAudio(TrumpetNoteSample.G_SHARP_4)
        case "A4":
            playAudio(TrumpetNoteSample.A_4)
        case "A#4":
            playAudio(TrumpetNoteSample.A_SHARP_4)
        case "B4":
            playAudio(TrumpetNoteSample.B_4)
        case "C5":
            playAudio(TrumpetNoteSample.C_5)
        case "C#5":
            playAudio(TrumpetNoteSample.C_SHARP_5)
        case "D5":
            playAudio(TrumpetNoteSample.D_5)
        case "D#5":
            playAudio(TrumpetNoteSample.D_SHARP_5)
        case "E5":
            playAudio(TrumpetNoteSample.E_5)
        case "F5":
            playAudio(TrumpetNoteSample.F_5)
        case "F#5":
            playAudio(TrumpetNoteSample.F_SHARP_5)
        case "G5":
            playAudio(TrumpetNoteSample.G_5)
        case "G#5":
            playAudio(TrumpetNoteSample.G_SHARP_5)
        case "A5":
            playAudio(TrumpetNoteSample.A_5)
        case "A#5":
            playAudio(TrumpetNoteSample.A_SHARP_5)
        case "None":
            mixer.music.stop()

# FACIAL RECOGNITION
mp_face = face_mesh
face = mp_face.FaceMesh(refine_landmarks=True)

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
    lip_right_x,

    dist_to_screen
):
    
    scaling_factor = STANDARD_DISTANCE / dist_to_screen

    print("scaling_factor:", scaling_factor)
    
    y_diff_outer_center_lip = abs(lip_center_outer_upper_y - lip_center_outer_lower_y)

    x_diff_edge_lip = abs(lip_left_x - lip_right_x)

    # print("lip_center_outer_upper_y: ", lip_center_outer_upper_y)
    # print("lip_center_outer_lower_y: ", lip_center_outer_lower_y)
    # print()
    print("diff: ", abs(lip_center_outer_upper_y - lip_center_outer_lower_y))

    print()

    print("lip_left_x: ", lip_left_x)
    print("lip_right_x: ", lip_right_x)

    print("x_diff: ", x_diff_edge_lip)
    print("y_diff_outer_center_lip: ", y_diff_outer_center_lip)

    lip_is_closed = math.ceil(lip_center_inner_upper_y * 100) >= math.floor(lip_center_inner_lower_y * 100)

    if not lip_is_closed:
        if x_diff_edge_lip <= 0.08 * scaling_factor:
            return "Pursed"

    # Closed lips
    if lip_is_closed:

        # Pursed lips, more air
        # 0.05 for mac
        if x_diff_edge_lip <= 0.08 * scaling_factor:
            return "Forced"
        
        elif y_diff_outer_center_lip <= 0.05 * scaling_factor:
            return "Strained"
        
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
# Forced

# C4: None + Closed
# C#4: All + Closed
# D4: BackFront + Closed
# D#4: MiddleFront + Closed
# E4: BackMiddle + Closed
# F4: Back + Closed
# F#4: Middle + Closed

# G4: None + Strained
# G#4: MiddleFront + Strained
# A4: BackMiddle + Strained
# A#4: Back + Strained
# B4: Middle + Strained

# C5: None + Pursed
# C#5: BackMiddle + Pursed
# D5: BackFront + Pursed
# D#5: MiddleFront + Pursed
# E5: Front + Pursed
# F5: Back + Pursed
# F#5: Middle + Pursed

# G5: None + Forced
# G#5: MiddleFront + Forced
# A5: BackMiddle + Forced
# A#5: Back + Forced
# B5: Middle + Forced # SOUND UNAVAILABLE

def notesClassification(valve_state, lip_state):
    
    if lip_state == "Closed":
        
        match valve_state:

            case "None":
                return "C4"
            
            case "BackMiddleFront":
                return "C#4"
            
            case "BackFront":
                return "D4"
            
            case "MiddleFront":
                return "D#4"
            
            case "BackMiddle":
                return "E4"
            
            case "Back":
                return "F4"
            
            case "Middle":
                return "F#4"
    
    elif lip_state == "Strained":

        match valve_state:

            case "None":
                return "G4"
        
            case "MiddleFront":
                return "G#4"
        
            case "BackMiddle":
                return "A4"
        
            case "Back":
                return "A#4"
        
            case "Middle":
                return "B4"
    
    elif lip_state == "Pursed":
        
        match valve_state:

            case "None":
                return "C5"
            
            case "BackMiddle":
                return "C#5"
            
            case "BackFront":
                return "D5"
            
            case "MiddleFront":
                return "D#5"
            
            case "Front":
                return "E5"
            
            case "Back":
                return "F5"
            
            case "Middle":
                return "F#5"

    elif lip_state == "Forced":

        match valve_state:

            case "None":
                return "G5"
        
            case "MiddleFront":
                return "G#5"
        
            case "BackMiddle":
                return "A5"
        
            case "Back":
                return "A#5"
        
            # case "Middle":
            #     return "B5"
    
    else:
        return "None"

notes_output = "Not Detected"
notes_output_prev = notes_output

def distFaceToScreen(
    left_iris_x,
    left_iris_y,
    right_iris_x,
    right_iris_y
):

    left_iris = [left_iris_x, left_iris_y]
    right_iris = [right_iris_x, right_iris_y]

    return 5 / math.dist(left_iris, right_iris)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    mixer.init()
 
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

            dist_face_to_screen = distFaceToScreen(
                left_iris_x=face_landmark.landmark[473].x,
                left_iris_y=face_landmark.landmark[473].y,
                right_iris_x=face_landmark.landmark[468].x,
                right_iris_y=face_landmark.landmark[468].y
            )

            print("dist_face_to_screen:", dist_face_to_screen)

            lip_state = airflowClassification(

                # Outer boundary of lips
                lip_center_outer_upper_y=face_landmark.landmark[0].y,
                lip_center_outer_lower_y=face_landmark.landmark[17].y,

                # Inner boundary of lips
                lip_center_inner_upper_y=face_landmark.landmark[13].y,
                lip_center_inner_lower_y=face_landmark.landmark[14].y,

                # Commmissures
                lip_left_x=face_landmark.landmark[291].x,
                lip_right_x=face_landmark.landmark[61].x,

                dist_to_screen=dist_face_to_screen
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

        # Detect change in

        notes_output = notesClassification(valve_state, lip_state)

        print("notes_output: ", notes_output)
        print("notes_output_prev: ",notes_output_prev)

        if notes_output_prev != notes_output:
            print("Note changed")
            playNote(notes_output)

        notes_output_prev = notes_output

        cv.putText(
            image_rgb, # image to draw text on
            notes_output, 
            (200, 500), # bottom left corner of text
            cv.FONT_HERSHEY_SIMPLEX, # font to use
            1, # font scale
            (255, 0, 0), # color
            2, # line thickness
        )
    
    else:
        mixer.music.stop()

    # Convert the RGB image back to BGR
    image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    
    # Display the resulting frame
    cv.imshow('Air Trumpet v1.0', image)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()