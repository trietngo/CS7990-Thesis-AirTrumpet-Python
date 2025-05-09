# Created by Triet Ngo for CS 7990: Master's Thesis
# at Northeastern University. Last updated: May 8th, 2025.

# IMPORT NECESSARY PACKAGES AND LIBRARIES
import enum
from pygame import mixer
import cv2 as cv
import mediapipe as mp
import math
from mediapipe.python.solutions import hands, drawing_styles
from mediapipe.python.solutions import face_mesh
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from datetime import datetime

# GLOBAL THRESHOLDS FOR LIP LANDMARK DETECTIONS

STANDARD_DISTANCE = 40 # Distance from face to MacBook Pro 16 2023 webcam is 40cm
SCREEN_TO_IRL_CONST = 5 
PURSED_LIPS_THRESHOLD_X = 0.08
STRAINED_LIPS_THRESHOLD_X = 0.1
STRAINED_LIPS_THRESHOLD_Y = 0.05

# TEST MODULE FOR IN-PERSON STUDY
notes_num_detected = 0
notes_tried_before_current = 0
notes_required = ["F#3", "C4", "C#4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "E5", "A#5"]
notes_required_matched = []
notes_required_remaining = ["F#3", "C4", "C#4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "E5", "A#5"]
notes_required_tries = []

class TrumpetNoteSample(enum.StrEnum):

    """Enumerate sound sample path of 31 possible trumpet tones.
    Audio truncated and processed using Audacity.
    """

    F_SHARP_3 = "data/sample_sounds_truncated/357378__mtg__trumpet-fsharp3-truncated.wav"
    G_3 = "data/sample_sounds_truncated/357568__mtg__trumpet-g3-truncated.wav"
    G_SHARP_3 = "data/sample_sounds_truncated/357566__mtg__trumpet-gsharp3-truncated.wav"
    A_3 = "data/sample_sounds_truncated/357380__mtg__trumpet-a3-truncated.wav"
    A_SHARP_3 = "data/sample_sounds_truncated/357589__mtg__trumpet-asharp3-truncated.wav"
    B_3 = "data/sample_sounds_truncated/357381__mtg__trumpet-b3-truncated.wav"
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

    # No quality sound samples for B5 and C6
    # B_5 = ""
    # C_6 = ""

def playAudio(path):

    """Play audio using PyAudio.
    This function loads the sound sample from the TrumpetNoteSample
    class and play the sample in a loop at volume of 1x system volume.

    Keyword arguments:
    path -- path to sound sample from TrumpetNoteSample class
    """

    mixer.music.load(path)
    mixer.music.set_volume(1)
    mixer.music.play(loops=-1, start=0)

def playNote(note):

    """Play the correct note using system audio output.

    Keyword arguments:
    note -- note to be played using playAudio()
    """

    match note:
        case "F#3":
            playAudio(TrumpetNoteSample.F_SHARP_3)
        case "G3":
            playAudio(TrumpetNoteSample.G_3)
        case "G#3":
            playAudio(TrumpetNoteSample.G_SHARP_3)
        case "A3":
            playAudio(TrumpetNoteSample.A_3)
        case "A#3":
            playAudio(TrumpetNoteSample.A_SHARP_3)
        case "B3":
            playAudio(TrumpetNoteSample.B_3)
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
        
        # No sound samples for B5 and C6

        # case "B5":
        #     playAudio(TrumpetNoteSample.B_5)
        # case "C6":
        #     playAudio(TrumpetNoteSample.C_6)

        # If no notes are detected
        # stop playing sounds
        case "None":
            mixer.music.stop()

# FACIAL RECOGNITION
mp_face = face_mesh
face = mp_face.FaceMesh(refine_landmarks=True) # Set refine-landmarks to True to include irises

# HAND GESTURE RECOGNITION
mp_hands = hands
hand = mp_hands.Hands()

# Initialize drawing utils for the hand and facial landmarks on current frame
mp_drawing = mp.solutions.drawing_utils
mp_drawing_lips = mp.solutions.drawing_utils

mp_drawing_styles = drawing_styles

custom_style_hands = mp_drawing_styles.get_default_hand_landmarks_style()
custom_connections_hands = list(mp_hands.HAND_CONNECTIONS)

custom_style_lips = mp_drawing_styles.get_default_face_mesh_tesselation_style()
custom_connections_lips = list(mp_face.FACEMESH_LIPS)

# LIP LANDMARKS ISOLATION (NOT WORKING)

# # Extract all lip landmarks
# lip_landmarks_2d = [list(connection_tuple) for connection_tuple in custom_connections_lips]

# # Flatten the lip landmarks
# lip_landmarks = [
#     lip_landmark
#     for pair in lip_landmarks_2d
#     for lip_landmark in pair
# ]

# # Remove duplicates
# lip_landmarks = list(set(lip_landmarks))

# # Remove non-lip feature drawing
# for lip_landmark_tuple in custom_connections_lips:
#     custom_style_lips[lip_landmark_tuple] = DrawingSpec(color=(0,0,0), thickness=None)
#     print(custom_style_lips[lip_landmark_tuple])

# for style in custom_style_lips:
#     print(style)

# EXCLUDE ALL THUMB, PINKY FINGER AND WRIST LANDMARKS
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

    # Change excluded landmarks color to white(0,0,0) with no thickness
    custom_style_hands[hand_landmark] = DrawingSpec(color=(0,0,0), thickness=None)

    # Remove all connections among the excluded landmarks
    custom_connections_hands = [
        connection_tuple
        for connection_tuple in custom_connections_hands 
        if hand_landmark.value not in connection_tuple
    ]

# PRINT DEBUGGING FOR EXCLUDED LANDMARKS

# print("Custom style hands: ")
# print(custom_style_hands)

# print("Custom connection hands: ")
# print(custom_connections_hands)

# print("Custom style lips: ")
# print(custom_style_lips)

# print("Custom connection lips after fix: ")
# print(custom_connections_lips)

# VIDEO CAPTURE INITIALIZATION
cap = cv.VideoCapture(0)
if not cap.isOpened():

    # If camera is not detected, exit the application.
    print("Cannot open camera.")
    exit()


def airflowClassification(

    # Lip Landmarks
    lip_center_outer_upper_y,
    lip_center_outer_lower_y,

    lip_center_inner_upper_y,
    lip_center_inner_lower_y,

    lip_edge_left_x,
    lip_edge_right_x,

    dist_to_screen
):
    
    """Classify different lip gestures to represent different
    levels of airflow strength.

    Keyword arguments:
    lip_center_outer_upper_y -- y-coord of upper outer center lip landmark, numbered 0
    lip_center_outer_lower_y -- y-coord of lower outer center lip landmark, numbered 17

    lip_center_inner_upper_y -- y-coord of upper inner center lip landmark, numbered 13
    lip_center_inner_lower_y -- y-coord of lower inner center lip landmark, numbered 14

    lip_edge_left_x -- x-coord of left commissure, numbered 291
    lip_edge_right_x -- x-coord of right commissure, numbered 61

    dist_to_screen -- standard distance from face to screen, default value is 40
    """
    
    # Scaling factor that keeps certain oral gesture thresholds
    # consistent regardless of distance from the user's face to
    # the screen
    scaling_factor = STANDARD_DISTANCE / dist_to_screen

    # print("scaling_factor:", scaling_factor)
    
    # y-axis distance between the outer center lip landmarks #0 and #17
    y_diff_outer_center_lip = abs(lip_center_outer_upper_y - lip_center_outer_lower_y)

    x_diff_edge_lip = abs(lip_edge_left_x - lip_edge_right_x)

    # print("lip_center_outer_upper_y: ", lip_center_outer_upper_y)
    # print("lip_center_outer_lower_y: ", lip_center_outer_lower_y)
    # print()
    # print("diff: ", abs(lip_center_outer_upper_y - lip_center_outer_lower_y))

    # print()

    # print("lip_left_x: ", lip_edge_left_x)
    # print("lip_right_x: ", lip_edge_right_x)

    # print("x_diff: ", x_diff_edge_lip)
    # print("y_diff_outer_center_lip: ", y_diff_outer_center_lip)

    # Lip is considered closed when the inner center lip landmarks
    # #13 and #14 meet or overlap on the y-axis
    lip_is_closed = math.ceil(lip_center_inner_upper_y * 100) >= math.floor(lip_center_inner_lower_y * 100)

    # If the lips are not closed
    # If the x-axis distance between two commissures are less
    # than the threshold times the scaling factor, register
    # a "Pursed" gesture
    if not lip_is_closed:
        if x_diff_edge_lip <= PURSED_LIPS_THRESHOLD_X * scaling_factor:
            return "Pursed"

    # If the lips are closed
    if lip_is_closed:

        # Pursed lips, more air
        # If the x-axis distance between two commissures is less
        # than the threshold times the scaling factor, register
        # a "Forced" gesture
        if x_diff_edge_lip <= PURSED_LIPS_THRESHOLD_X * scaling_factor:
            return "Forced"
        
        # If the y-axis distance between the outer center landmarks
        # is less than the y-axis threshold for strained lips
        # and if the x-axis distance between two commissures is greater
        # than the threshold for pursed lips
        elif y_diff_outer_center_lip <= STRAINED_LIPS_THRESHOLD_Y * scaling_factor and x_diff_edge_lip > PURSED_LIPS_THRESHOLD_X * scaling_factor:

            # If the x-axis distance between two commissures is greater
            # or equal to the strained lips threshold,
            # register a "Strained" gesture
            if x_diff_edge_lip >= STRAINED_LIPS_THRESHOLD_X * scaling_factor:
                return "Strained"
            
            # If the x-axis distance between two commissures is less than
            # the strained lips threshold, register a "Tensed" gesture
            elif x_diff_edge_lip < STRAINED_LIPS_THRESHOLD_X * scaling_factor:
                return "Tensed"
        
        # If the lips are closed, and no other thresholds are satisfied
        # register a "Closed" gesture
        else:
            return "Closed"

    # If no other criteria are met
    # register an "Open" gesture
    else:
        return "Open"


def valveClassification(
        
    # Hand Landmarks
    index_pip_y,
    index_tip_y,
    middle_pip_y,
    middle_tip_y,
    ring_pip_y,
    ring_tip_y,

):

    """Classify different hand gestures to represent a trumpet
    valve press.

    Keyword arguments:
    index_pip_y -- y-coord of the index finger's PIP
    index_tip_y -- y-coord of the index finger tip
    middle_pip_y -- y-coord of the middle finger's PIP
    middle_tip_y -- y-coord of the middle finger tip
    ring_pip_y -- y-coord of the ring finger's PIP
    ring_tip_y -- y-coord of the ring finger tip
    """


    def isBackValvePressed():

        """
        Function to check if the front valve is pressed
        if the y-coord of the index tip is lower than the y-coord of the index pip
        then register a press and return True
        """

        # Use greater than because the coords are inverted
        if index_tip_y >= index_pip_y:
            return True
        
        else:
            return False


    def isMiddleValvePressed():

        """
        Function to check if the middle valve is pressed
        if the y-coord of the middle finger tip is lower than the y-coord of the middle finger pip
        then register a press and return True
        """

        # Use greater than because the coords are inverted
        if middle_tip_y >= middle_pip_y:
            return True
        
        else:
            return False

    
    def isFrontValvePressed():

        """
        Function to check if the back valve is pressed
        if the y-coord of the ring finger tip is lower than the y-coord of the ring finger pip
        then register a press and return True
        """

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

    # Valve state initialized to "None", meaning no valves are pressed
    valve_state = "None"

    # If a valve is pressed, initialize valve_state to empty string ""
    if isBackValvePressed() or isMiddleValvePressed() or isFrontValvePressed():
        valve_state = ""

    # If back valve is pressed, append "Back" to valve_state
    if isBackValvePressed():
        valve_state += "Back"
    
    # If middle valve is pressed, append "Middle" to valve_state
    if isMiddleValvePressed():
        valve_state += "Middle"
    
    # If front valve is pressed, append "Front" to valve_state
    if isFrontValvePressed():
        valve_state += "Front"
    
    # Return the valve state
    return valve_state


def toneClassification(valve_state, lip_state):

    """Return the tone matching detected hand and the oral gestures

    Playable tones:

    F#3: All + Closed
    G3: BackFront + Closed
    G#3: MiddleFront + Closed
    A3: BackMiddle + Closed
    A#3: Back + Closed
    B3: Middle + Closed
    C4: None + Closed

    C#4: All + Tensed
    D4: BackFront + Tensed
    D#4: MiddleFront + Tensed
    E4: BackMiddle + Tensed
    F4: Back + Tensed
    F#4: Middle + Tensed
    G4: None + Tensed

    G#4: MiddleFront + Strained
    A4: BackMiddle + Strained
    A#4: Back + Strained
    B4: Middle + Strained
    C5: None + Strained

    C#5: BackMiddle + Pursed
    D5: BackFront + Pursed
    D#5: MiddleFront + Pursed
    E5: Front + Pursed
    F5: Back + Pursed
    F#5: Middle + Pursed
    G5: None + Pursed

    G#5: MiddleFront + Forced
    A5: BackMiddle + Forced
    A#5: Back + Forced
    B5: Middle + Forced # SOUND UNAVAILABLE
    C6: None + Forced # SOUND UNAVAILABLE

    Keyword arguments:
    valve_state -- hand gesture with regard to valve presses
    lip_state -- oral gesture representing airflow strength
    """
    
    if lip_state == "Closed":
        
        match valve_state:
            
            case "BackMiddleFront":
                return "F#3"
            
            case "BackFront":
                return "G3"
            
            case "MiddleFront":
                return "G#3"
            
            case "BackMiddle":
                return "A3"
            
            case "Front":
                return "A3"
            
            case "Back":
                return "A#3"
            
            case "Middle":
                return "B3"
            
            case "None":
                return "C4"
    
    elif lip_state == "Tensed":

        match valve_state:

            case "BackMiddleFront":
                return "C#4"
            
            case "BackFront":
                return "D4"
            
            case "MiddleFront":
                return "D#4"
            
            case "BackMiddle":
                return "E4"
            
            case "Front":
                return "E4"
            
            case "Back":
                return "F4"
            
            case "Middle":
                return "F#4"

            case "None":
                return "G4"

    elif lip_state == "Strained":

        match valve_state:
        
            case "MiddleFront":
                return "G#4"
        
            case "BackMiddle":
                return "A4"
            
            case "Front":
                return "A4"
        
            case "Back":
                return "A#4"
        
            case "Middle":
                return "B4"
            
            case "None":
                return "C5"
    
    elif lip_state == "Pursed":
        
        match valve_state:
            
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
            
            case "None":
                return "G5"

    elif lip_state == "Forced":

        match valve_state:
        
            case "MiddleFront":
                return "G#5"
        
            case "BackMiddle":
                return "A5"
            
            case "Front":
                return "A5"
        
            case "Back":
                return "A#5"
            
            case "Middle":
                return "B5"
            
            case "None":
                return "C6"
    
    else:
        return "None"


def distFaceToScreen(
    left_iris_x,
    left_iris_y,
    right_iris_x,
    right_iris_y
):

    """Calculates the distance between the user's irises
    and roughly abstract that value to the distance between
    the user's face and the screen.

    Keyword arguments:
    left_iris_x -- Left iris' x-coord
    left_iris_y -- Left iris' y-coord
    right_iris_x -- Right iris' x-coord
    right_iris_y -- Right iris' y-coord
    """

    # Left and right irises' 2D coordinates
    left_iris = [left_iris_x, left_iris_y]
    right_iris = [right_iris_x, right_iris_y]

    # Calculate the distance from face to screen
    return SCREEN_TO_IRL_CONST / math.dist(left_iris, right_iris)

# Initialize current date and time
# For in-person study
start_datetime = datetime.now()

# Initialize tone output
# and set tone output in the previous frame to be the same as the current frame
tone_output = "Not Detected"
tone_output_prev = tone_output

# While a frame is captured
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Initialize PyGame Mixer for sound output
    mixer.init()
 
    # If frame is read correctly, ret is True
    # if ret is not True, meaning current frame cannot be captured
    # break and exit the program
    if not ret:
        print("Frame cannot be read. Exiting ...")
        break

    # Convert current frame to RGB
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect gestures using MediaPipe hand and face detection models
    # Current frame is marked as not writeable to pass by reference
    # then remarked as writable to improve performance
    image_rgb.flags.writeable = False
    results_hands = hand.process(image_rgb)
    results_face = face.process(image_rgb)
    image_rgb.flags.writeable = True

    # Initialize valve_state and lip_state as "Not Detected"
    # to detect change later
    valve_state = "Not Detected"
    lip_state = "Not Detected"

    # If hand landmarks are detected
    if results_hands.multi_hand_landmarks:

        # For each landmark in the detected landmarks
        for hand_landmarks in results_hands.multi_hand_landmarks:

            # Classify the current hand gestures into valve states
            # using relevant landmarks: 6, 8, 10, 12, 14, 16
            valve_state = valveClassification(
                
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

            # Draw text on frame
            cv.putText(
                image_rgb, # image on which to draw text
                valve_state, # current valve_state string
                (200, 600), # bottom left corner of text
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                1, # font scale
                (255, 0, 0), # color
                2, # line thickness
            )

            # Draw landmarks on the detected hands
            mp_drawing.draw_landmarks(
                image_rgb, # image on which to draw
                hand_landmarks, # detected landmarks
                custom_connections_hands, # custom connections
                custom_style_hands # custom hands
                # mp_hands.HAND_CONNECTIONS
            )
    
    # If face landmarks are detected
    if results_face.multi_face_landmarks:

        # For each landmark in the detected landmarks
        for face_landmark in results_face.multi_face_landmarks:

            # Calculate distance from face to screen
            # using iris landmarks
            dist_face_to_screen = distFaceToScreen(
                left_iris_x=face_landmark.landmark[473].x,
                left_iris_y=face_landmark.landmark[473].y,
                right_iris_x=face_landmark.landmark[468].x,
                right_iris_y=face_landmark.landmark[468].y
            )

            # print("dist_face_to_screen:", dist_face_to_screen)

            # Classify oral gestures using lip landmarks
            lip_state = airflowClassification(

                # Outer boundary of lips
                lip_center_outer_upper_y=face_landmark.landmark[0].y,
                lip_center_outer_lower_y=face_landmark.landmark[17].y,

                # Inner boundary of lips
                lip_center_inner_upper_y=face_landmark.landmark[13].y,
                lip_center_inner_lower_y=face_landmark.landmark[14].y,

                # Commmissures
                lip_edge_left_x=face_landmark.landmark[291].x,
                lip_edge_right_x=face_landmark.landmark[61].x,

                dist_to_screen=dist_face_to_screen
            )

            # Draw text
            cv.putText(
                image_rgb, # frame to draw text on
                lip_state, # classified oral gesture
                (200, 650), # text coordinate
                cv.FONT_HERSHEY_SIMPLEX, # font to use
                1, # font scale
                (255, 0, 0), # color
                2, # line thickness
            )

            # Draw landmarks
            mp_drawing_lips.draw_landmarks(
                image_rgb, # frame to draw text on
                face_landmark, # default MediaPipe face landmarks
                # connections=custom_connections_lips,
                landmark_drawing_spec=custom_style_lips # landmark styling
                # mp_face.FACEMESH_CONTOURS
            )

    # If both hand and oral gestures are detected
    if results_hands.multi_hand_landmarks and results_face.multi_face_landmarks:

        # Detect change in tone

        # Classify current tone output
        tone_output = toneClassification(valve_state, lip_state)

        # print("tone_output: ", tone_output)
        # print("tone_output_prev: ",tone_output_prev)

        # If the tone in the previous frame is different
        # from the tone in the current frame
        # there is a change in tone, and Air Trumpet
        # should play a new tone
        if tone_output_prev != tone_output:
            print("Note changed")
            playNote(tone_output)

            # TESTING MODULE
            # If the tone output is not "None"
            if tone_output != "None":
                # print("Add a note")

                # Increase the number of detected tones
                # Increase the number of tones tried before current tone
                notes_num_detected = notes_num_detected + 1;
                notes_tried_before_current = notes_tried_before_current + 1;

                # If there are still required notes to be played
                # and the tone output matches the first tone in the required list
                if len(notes_required_remaining) > 0 and tone_output == notes_required_remaining[0]:
                    
                    # Add the tone to the list of matched tones
                    notes_required_matched.append(tone_output)

                    # Add the number of tries to the list of number of tries for each required tone
                    notes_required_tries.append(notes_tried_before_current)

                    # Reset the number of tries to 0 to get ready for the next required tone
                    notes_tried_before_current = 0

                    # Remove the first tone in the list of remaining required tones
                    # which is the tone that was correctly played
                    notes_required_remaining.pop(0)

        # Update the tone in previous frame to current frame
        tone_output_prev = tone_output

        # Draw text
        cv.putText(
            image_rgb, # image to draw text on
            tone_output, 
            (200, 700), # bottom left corner of text
            cv.FONT_HERSHEY_SIMPLEX, # font to use
            1, # font scale
            (255, 0, 0), # color
            2, # line thickness
        )
    
    # Else if no hand or oral gestures are detected
    # mixer stops outputting sounds
    else:
        mixer.music.stop()
    
    # On-screen texts for testing
    cv.putText(
        image_rgb, # image to draw text on
        "Required notes: " + str(notes_required), 
        (200, 750), # bottom left corner of text
        cv.FONT_HERSHEY_SIMPLEX, # font to use
        1, # font scale
        (255, 0, 0), # color
        2, # line thickness
    )

    cv.putText(
        image_rgb, # image to draw text on
        "Matched notes: " + str(notes_required_matched), 
        (200, 800), # bottom left corner of text
        cv.FONT_HERSHEY_SIMPLEX, # font to use
        1, # font scale
        (255, 0, 0), # color
        2, # line thickness
    )

    cv.putText(
        image_rgb, # image to draw text on
        "Remaining notes: " + str(notes_required_remaining), 
        (200, 850), # bottom left corner of text
        cv.FONT_HERSHEY_SIMPLEX, # font to use
        1, # font scale
        (255, 0, 0), # color
        2, # line thickness
    )

    # Convert the RGB image back to BGR
    image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
    
    # Display the resulting frame
    cv.imshow('Air Trumpet v1.0', image)

    # If user press q, exit the program
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()

# Debugging screen for in-person testing
print()

print("Start date and time:", start_datetime)

end_datetime = datetime.now()
print("Completion date and time:", end_datetime)

print()
print("Number of notes detected:", notes_num_detected)

if notes_num_detected > 0:
    accuracy = len(notes_required_matched) / notes_num_detected
    print("Accuracy:", round(accuracy, 2))

print(notes_required_remaining)
print("Matched notes:", notes_required_matched)
print("Number of tries for each note:", notes_required_tries)
