"""
Basic Yawn Detection System with Alert
Restoring basic yawn detection functionality with simple alert system
"""
import cv2
import mediapipe as mp
import math
import threading
import time
import pygame
from config import *

# Initialize pygame mixer for sound control
pygame.mixer.init()

# Global variables for yawn detection state
is_yawning = False
yawn_counter = 0
sound_playing = False
consecutive_frames_threshold = CONSECUTIVE_FRAMES_THRESHOLD
alert_sound = None

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

def euclidean_distance(pt1, pt2):
    """Calculate euclidean distance between two points"""
    return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

def load_alert_sound():
    """Load the alert sound file"""
    global alert_sound
    try:
        alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
    except Exception as e:
        print(f"Error loading sound file: {e}")
        alert_sound = None

def play_alert_sound():
    """Play alert sound with loop"""
    global sound_playing
    if alert_sound and not sound_playing:
        sound_playing = True
        alert_sound.play(-1)  # -1 means loop indefinitely
        print("Alert sound started")

def stop_alert_sound():
    """Stop the alert sound immediately"""
    global sound_playing
    if sound_playing:
        pygame.mixer.stop()
        sound_playing = False
        print("Alert sound stopped")

def main():
    """Main application for yawn detection with continuous sound control"""
    global is_yawning, yawn_counter, sound_playing

    # Load the alert sound at startup
    load_alert_sound()

    print("Starting Yawn Detection System with Continuous Sound Control...")
    print("Press ESC to exit")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            current_yawn_detected = False

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # Upper and lower lip points (based on Mediapipe face mesh indices)
                    upper_lip = landmarks.landmark[13]  # upper inner lip
                    lower_lip = landmarks.landmark[14]  # lower inner lip

                    mouth_open = euclidean_distance(upper_lip, lower_lip)

                    # Heuristic threshold: adjust if needed
                    if mouth_open > YAWN_THRESHOLD:
                        current_yawn_detected = True
                        yawn_counter += 1

                        # Display yawn detected message
                        cv2.putText(frame, "YAWN DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)

                        # If this is the start of a yawn (not already yawning)
                        if not is_yawning and yawn_counter >= consecutive_frames_threshold:
                            is_yawning = True
                            play_alert_sound()
                            print("Yawn started - Alert triggered!")
                    else:
                        # Reset yawn counter if mouth is not open
                        yawn_counter = 0

            # If no yawn detected this frame, reset the yawning state and stop sound
            if not current_yawn_detected:
                if is_yawning:
                    print("Yawn ended")
                    stop_alert_sound()  # Stop the sound immediately
                is_yawning = False
                yawn_counter = 0

            # Display yawn status on screen
            status = "YAWNING" if is_yawning else "NORMAL"
            cv2.putText(frame, f"Status: {status}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if not is_yawning else (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Yawn Detector", frame)

            # Check for ESC key press to exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stop_alert_sound()  # Make sure sound is stopped
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("Yawn Detection System shutdown complete.")

if __name__ == "__main__":
    main()
