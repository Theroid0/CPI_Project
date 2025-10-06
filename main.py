import cv2
import time
import math
import threading
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
import mediapipe as mp
import winsound   
import XInput
from config import *

alarm_playing = False
controller_vibrating = False
is_yawning = False
yawn_counter = 0
sound_playing = False
consecutive_frames_threshold = CONSECUTIVE_FRAMES_THRESHOLD

def euclidean_distance(pt1, pt2):
    """Calculate euclidean distance between two points"""
    return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

def play_alarm(sound_file="warning.wav"):
    global alarm_playing
    alarm_playing = True
    try:
        winsound.PlaySound(sound_file, winsound.SND_FILENAME)
    except Exception as e:
        print("Error playing alarm:", e)
    alarm_playing = False

def vibrate_controller(user_index, duration=2):
    global controller_vibrating
    controller_vibrating = True
    try:
        XInput.set_vibration(user_index, 0.5, 0.5)
        time.sleep(duration)
        XInput.set_vibration(user_index, 0, 0)
    except Exception as e:
        print("Vibration error:", e)
    controller_vibrating = False

def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])  # vertical 1
    B = dist.euclidean(eye_points[2], eye_points[4])  # vertical 2
    C = dist.euclidean(eye_points[0], eye_points[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]  
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
window_size = 10

def main():
    global alarm_playing,is_yawning,yawn_counter
    cap = cv2.VideoCapture(0)

    counter = 0
    distracted = False
    holdingSteering = True
    queue = deque(maxlen=max(1, window_size))
    connected = XInput.get_connected()

    try:
        while True:
            if(connected[0]):
                state = XInput.get_state(0)
                pressed = XInput.get_button_values(state=state)
                if not pressed["RIGHT_SHOULDER"]:
                    holdingSteering = False
                elif pressed["RIGHT_SHOULDER"]:
                    holdingSteering = True
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            current_yawn_detected = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    left_eye = [(int(face_landmarks.landmark[i].x * w),
                                 int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                    right_eye = [(int(face_landmarks.landmark[i].x * w),
                                  int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
                    upper_lip = face_landmarks.landmark[13]  # upper inner lip
                    lower_lip = face_landmarks.landmark[14]  # lower inner lip

                    mouth_open = euclidean_distance(upper_lip, lower_lip)
                    if mouth_open > YAWN_THRESHOLD:
                        current_yawn_detected = True
                        yawn_counter += 1
                        if not is_yawning and yawn_counter >= consecutive_frames_threshold:
                            is_yawning = True
                            print("Yawn started - Alert triggered!")
                    else:
                        # Reset yawn counter if mouth is not open
                        yawn_counter = 0
                        
                    if not current_yawn_detected:
                        if is_yawning:
                            print("Yawn ended")
                        is_yawning = False
                        yawn_counter = 0

                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)
                    ear = (leftEAR + rightEAR) / 2.0

                    queue.append(ear)
                    smoothed_ear = np.mean(queue)

                    cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0,255,0), 1)
                    cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0,255,0), 1)

                    cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                    if smoothed_ear < EAR_THRESHOLD:
                        counter += 1
                        cv2.putText(frame, f"Closed Frames: {counter}", (10,70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                        if counter >= EAR_CONSEC_FRAMES:
                            distracted = True
                    else:
                        counter = 0
                        distracted = False
                        
                    if is_yawning or distracted or not holdingSteering:
                        if is_yawning:
                            cv2.putText(frame, "YAWN DETECTED!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
                        if distracted:
                            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (10,120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        if not holdingSteering:
                            cv2.putText(frame, "!!! HOLD THE STEERING WHEEL !!!", (10,450),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        if not alarm_playing:
                            threading.Thread(target=play_alarm,
                                             args=("warning.wav",), daemon=True).start()
                        if connected[0] and not controller_vibrating:
                            threading.Thread(target=vibrate_controller, args=(0,), daemon=True).start()

            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
